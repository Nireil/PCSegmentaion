import logging
import numpy as np
import open3d as o3d
import os
from os.path import join, exists
from .base_dataset import Basedataset, BasedatasetSplit

log = logging.getLogger(__name__) 

class Toronto3D(Basedataset):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.label_to_names = self.get_label_to_names()
        for i in cfg.ignored_label_inds:
            del self.label_to_names[i]
        # self.label_keys = np.sort([k for k, v in self.label_to_names.items()])#
        # self.label_to_idx = {l: i for i, l in enumerate(self.label_keys)} # 作用不明

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'Unclassified',
            1: 'Ground',
            2: 'Road_markings',
            3: 'Natural',
            4: 'Building',
            5: 'Utility_line',
            6: 'Pole',
            7: 'Car',
            8: 'Fence'
        }
        return label_to_names
    
    def get_split(self, split):
        return Toronto3DSplit(self, split=split)

    def get_split_list(self, split):
        if split == 'test':
            files = self.cfg.test_files
        elif split == 'train':
            files = self.cfg.train_files
        elif split == 'valid':
            files = self.cfg.valid_files
        elif split == 'all':
            files = self.cfg.valid_files + self.cfg.train_files + self.cfg.test_files
        else:
            raise ValueError("Invalid split {}".format(split))
        return files

    def get_pointnum(self):
        num_pc_files = len(self.get_split_list('all'))
        num_all_classes = len(self.label_to_names)  # 包括被忽略的类别
        numpts_perfile_perclass = np.zeros((num_pc_files, num_all_classes), dtype=np.int32)
        pc_path = self.cfg.path
        pc_num = 0
        for root, dirs, files in os.walk(pc_path):
            for name in files:
                if name.endswith('.ply'):
                    data = o3d.t.io.read_point_cloud(join(root,name)).point
                    labels = data['scalar_Label'].numpy().astype(np.int32).reshape((-1,))
                    labels, counts = np.unique(labels, return_counts=True)
                    for i, label in enumerate(labels):
                        numpts_perfile_perclass[pc_num][label] += counts[i]
                    pc_num += 1
        num_per_class = np.sum(numpts_perfile_perclass,0)
        num_per_class = np.delete(num_per_class, self.cfg.ignored_label_inds, axis=0)
        return num_per_class

    def is_tested(self, path, attr):
        name = attr['name']
        store_path = join(path, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, path, predict_labels, attr):
        if not exists(path):
            os.makedirs(path)
        name = attr['name'][0]
        pred = predict_labels
        pred = pred.cpu().data.numpy()
        for ign in self.cfg.ignored_label_inds:
            pred[pred >= ign] += 1
        store_path = join(path, name + '.npy')
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))

class Toronto3DSplit(BasedatasetSplit):

    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list), split))
        self.UTM_OFFSET = [627285, 4841948, 0]
    
    def __len__(self):
        return len(self.path_list)

    def get_data(self, cloud_id): # 已经划分了split
        '''
        读取序号为cloud_id的原始点云文件
        '''                    
        pointcloud_path = self.cfg.path + '/' + self.path_list[cloud_id]
        log.debug("get data called {}".format(self.path_list[cloud_id]))
        data = o3d.t.io.read_point_cloud(pointcloud_path).point 
        points = data["positions"].numpy() - self.UTM_OFFSET
        points = np.float32(points)
        colors = data["colors"].numpy().astype(np.float32)
        intensities = data["scalar_Intensity"].numpy().astype(np.float32)
        features = np.concatenate((colors, intensities), axis=1)
        labels = data['scalar_Label'].numpy().astype(np.int32).reshape((-1,))
        data = {'points':points, 'features':features, 'labels':labels}
        # inputs = {'data':data, 'attr':self.get_attr(cloud_id)}
        # return inputs
        return data

    def get_attr(self, cloud_id):
        name = self.path_list[cloud_id].split('.')[0]
        pc_path = self.cfg.path + '/' + self.path_list[cloud_id]
        split = self.split
        attr = {'cloud_id':cloud_id, 'name':name, 'pc_path':pc_path, 'split':split}
        return attr