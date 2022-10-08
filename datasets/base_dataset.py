from abc import ABC, abstractmethod
import numpy as np
import glob, random, logging
from os import makedirs
from os.path import join, exists
from requests import delete
from sklearn.neighbors import KDTree

from .utils.augmentation import SemsegAugmentation
from .utils.dataprocessing import DataProcessing as DP
from .utils.points_sampler import spatially_regular_sampler, random_sampler, class_balanced_sampler
log = logging.getLogger(__name__)

class Basedataset(ABC):

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg # 此cfg只是cfg.dataset
        self.rng = np.random.default_rng(kwargs.get('seed', None))
    
    @abstractmethod
    def get_pointnum(self):
        """Returns the number of point for every class."""
        return 

    @abstractmethod
    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return

    @abstractmethod
    def get_split_list(self, split):
        """Returns the pointcloud list for the given split."""
        return

    @abstractmethod
    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        return False
 
    @abstractmethod
    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        return

class BasedatasetSplit(ABC):
    def __init__(self, dataset, split):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        self.path_list = path_list
        # self.length = len(path_list)
        self.split = split
        self.dataset = dataset

        if split == 'test':
            self.sampler = spatially_regular_sampler(self)
        else:
            if dataset.cfg.get_input.method == 'spatially_regular_sampler':
                self.sampler = spatially_regular_sampler(self)
            elif dataset.cfg.get_input.method == 'random_sampler':
                self.sampler = random_sampler(self)
            elif dataset.cfg.get_input.method == 'class_balanced_sampler':
                self.sampler = class_balanced_sampler(self)
            else:
                raise NotImplementedError(dataset.cfg.get_input.method)
        self.augmenter =  SemsegAugmentation(dataset.cfg.augment, seed = 42)

    @abstractmethod
    def __len__(self):
        """Returns the number of pointclouds in the split."""
        return 0
    
    @abstractmethod
    def get_data(self, idx):
        """Returns the data for the given index."""
        return {}

    def preprocess(self, data):
        """Returns the data after preprocess for the given index."""

        if self.cfg.preprocess.method == 'grid_subsampling':
            grid_size = self.cfg.preprocess.parameter['grid_size']
            sub_points, sub_features, sub_labels = DP.grid_subsampling(data['points'], data['features'], data['labels'], grid_size)
            search_tree = KDTree(sub_points)
            if self.split == 'test':
                proj_inds = np.squeeze(search_tree.query(data['points'], return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
            else:
                proj_inds = None
            data = {'points':sub_points, 'features':sub_features, 'labels':sub_labels, 'tree':search_tree, 'proj_inds': proj_inds}
        else:
            raise NotImplementedError(self.cfg.preprocess.method)

        return data

    def sampled_points(self, data, attr):
        cloud_id = attr['cloud_id']     # 输入attr导入cloud_id
        point_inds = self.sampler.get_sampled_points(data, cloud_id)
        input_points = data['points'][point_inds]

        if hasattr(self.cfg, 'add_intensity'):
            if self.cfg.add_intensity:
                if hasattr(self.cfg, 'no_rgb'):
                    if self.cfg.no_rgb:
                        input_features = np.delete(data['features'][point_inds],[0,1,2],axis=1)
                    else:
                        input_features = data['features'][point_inds]
                else:
                    input_features = data['features'][point_inds]
            else:
                if hasattr(self.cfg, 'no_rgb'):
                    if self.cfg.no_rgb:
                        input_features = None
                    else:
                        input_features = np.delete(data['features'][point_inds],[3],axis=1)
                else:
                    input_features = np.delete(data['features'][point_inds],[3],axis=1)
        else:
            if hasattr(self.cfg, 'no_rgb'):
                if self.cfg.no_rgb:
                    input_features = None
                else:
                    input_features = np.delete(data['features'][point_inds],[3],axis=1)
            else:
                input_features = np.delete(data['features'][point_inds],[3],axis=1)

        input_labels = data['labels'][point_inds]
        point_inds = point_inds if self.split == 'test' else None
        proj_inds = data['proj_inds'] if self.split == 'test' else None
        data = {'points':input_points, 'features':input_features, 'labels':input_labels,\
                'point_inds':point_inds, 'proj_inds': proj_inds}

        return data

    def augment(self, data):
        augment_cfg = self.cfg.augment
        val_augment_cfg = {}
        input_points, input_features, input_labels = data['points'], data['features'], data['labels']
        if 'recenter' in augment_cfg:
            val_augment_cfg['recenter'] = augment_cfg['recenter']
        if 'normalize' in augment_cfg:
            val_augment_cfg['normalize'] = augment_cfg['normalize']
        
        if self.split == 'train':
            self.augmenter.augment(input_points, input_features, input_labels, augment_cfg)
        else:
            self.augmenter.augment(input_points, input_features, input_labels, val_augment_cfg)
        # input_features = input_points.copy() if input_features is None  else input_features # 去冗余
        input_features = input_points.copy() if input_features is None  else np.concatenate([input_points, input_features], axis=1)
        
        # if hasattr(self.cfg, 'de_re'):
        #     if self.cfg.de_re:
        #         input_features = input_points.copy() if input_features is None  else input_features # 去冗余
        #     else:
        #          input_features = input_points.copy() if input_features is None  else np.concatenate([input_points, input_features], axis=1)
        # else:
        #     input_features = input_points.copy() if input_features is None  else np.concatenate([input_points, input_features], axis=1)
        
        data = {'points':input_points, 
                'features':input_features, 'labels':input_labels,\
                'point_inds':data['point_inds'], 'proj_inds':data['proj_inds']}
        # if input_features is None:
        #     del data['features']
        return data

    @abstractmethod
    def get_attr(self, idx):
        """Returns the attributes for the given index."""
        return {}
