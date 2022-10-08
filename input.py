import logging, glob, random
from cv2 import transform
import numpy as np
from tqdm import tqdm
from typing import Callable
from os import makedirs, listdir
from os.path import exists, join, splitext
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

class inputpoints(Dataset):                # 根据config确定输入方式，即如何__getitem__

    def __init__(self, 
                 cfg,
                 dataset=None,
                 preprocess=None,
                 use_cache=True,
                 transform=None,
                 **kwargs):
        self.dataset = dataset
        self.preprocess = preprocess
        self.split = dataset.split

        if self.split == 'train':
            self.num_per_epoch = cfg.train_steps_per_epoch* cfg.train_batch_size
        elif self.split == 'valid':
            self.num_per_epoch = cfg.valid_steps_per_epoch* cfg.valid_batch_size     
        else:
            self.num_per_epoch = None

        if preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg.preprocess, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'
            para = getattr(dataset.cfg.preprocess,'parameter')
            for i, key in enumerate(para.keys()):
                cache_key=key+'_'+str(para[key])
                if i+1<len(para.keys()):
                    cache_key += '+'
            self.cache_convert = Cache(preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=cache_key)
            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    name = attr['name']
                    if name in self.cache_convert.cached_ids:
                        continue
                    data = dataset.get_data(idx)
                    # cache the data
                    self.cache_convert(name, data)

        else:
            self.cache_convert = None
            
        self.transform = transform
        (dataset.sampler).initialize_with_dataset(self)

    def __getitem__(self, index):
        dataset=self.dataset
        index = index % len(dataset)
        attr = dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(dataset.get_data(index)) # open3d在此传入split,重构后不需要
        else:
            data = dataset.get_data(index)
        data = dataset.sampled_points(data, attr)   # 用attr传入cloud_id
        data = dataset.augment(data)
        if self.transform is not None:
            data = self.transform(data)
        inputs = {'data':data, 'attr':attr}
        return inputs

    def __len__(self):
        if self.num_per_epoch is not None:
            return self.num_per_epoch
        else:
            return len(self.dataset)


def make_dir(folder_name):
    """Create a directory.

    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)

class Cache(object):
    """Cache converter for preprocessed data."""

    def __init__(self, func: Callable, cache_dir: str, cache_key: str):
        """Initialize.

        Args:
            func: preprocess function of a model.
            cache_dir: directory to store the cache.
            cache_key: key of this cache
        Returns:
            class: The corresponding class.
        """
        self.func = func
        self.cache_dir = join(cache_dir, cache_key)
        make_dir(self.cache_dir)
        self.cached_ids = [splitext(p)[0] for p in listdir(self.cache_dir)]

    def __call__(self, unique_id: str, *data):
        """Call the converter. If the cache exists, load and return the cache,
        otherwise run the preprocess function and store the cache.

        Args:
            unique_id: A unique key of this data.
            data: Input to the preprocess function.

        Returns:
            class: Preprocessed (cache) data.
        """
        fpath = join(self.cache_dir, str('{}.npy'.format(unique_id)))

        if not exists(fpath):
            output = self.func(*data)

            self._write(output, fpath)
            self.cached_ids.append(unique_id)
        else:
            output = self._read(fpath)

        return self._read(fpath)

    def _write(self, x, fpath):
        np.save(fpath, x)

    def _read(self, fpath):
        return np.load(fpath, allow_pickle=True).item()

            


