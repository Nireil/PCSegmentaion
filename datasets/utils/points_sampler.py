import numpy as np
import random

class class_balanced_sampler(object):
    """Random sampler for semantic segmentation datasets."""

    def __init__(self, data_split): # train_split, valid_split
        self.data_split = data_split
        self.length = len(data_split)
        self.split = self.data_split.split

    def __len__(self):
        return self.length

    def initialize_with_dataset(self, dataset): # train_dataset, valid_dataset
        self.length = len(dataset)

    def get_cloud_sampler(self):
        def gen():
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i
        return gen()

    def get_sampled_points(self, data, cloud_id):
        cfg = self.data_split.cfg
        num_points = cfg.get_input.parameter['num_points']
        label_to_names = self.data_split.dataset.label_to_names
        valid_labels = list(label_to_names.keys())
        center_label = np.random.choice(valid_labels,1)
        label_idxs = np.where(data['labels']==center_label)[0].tolist()
        center_idx = np.random.choice(label_idxs, 1)
        center_point = data['points'][center_idx, :].reshape(1,-1)
        if (data['points'].shape[0] < num_points):
            diff = num_points - data['points'].shape[0]
            point_inds = np.array(range(data['points'].shape[0]))
            point_inds = list(point_inds) + list(random.choices(point_inds, k=diff))
            point_inds = np.asarray(point_inds)
        else:
            point_inds = data['tree'].query(center_point, k=num_points)[1][0]
        random.shuffle(point_inds)

        return point_inds

class random_sampler(object):
    """Random sampler for semantic segmentation datasets."""

    def __init__(self, data_split): # train_split, valid_split
        self.data_split = data_split
        self.length = len(data_split)
        self.split = self.data_split.split

    def __len__(self):
        return self.length

    def initialize_with_dataset(self, dataset): # train_dataset, valid_dataset
        self.length = len(dataset)

    def get_cloud_sampler(self):
        def gen():
            ids = np.random.permutation(self.length)
            for i in ids:
                yield i
        return gen()

    def get_sampled_points(self, data, cloud_id):
        cfg = self.data_split.cfg
        num_points = cfg.get_input.parameter['num_points']
        center_idx = np.random.choice(data['points'].shape[0], 1)
        center_point = data['points'][center_idx, :].reshape(1,-1)
        if (data['points'].shape[0] < num_points):
            diff = num_points - data['points'].shape[0]
            point_inds = np.array(range(data['points'].shape[0]))
            point_inds = list(point_inds) + list(random.choices(point_inds, k=diff))
            point_inds = np.asarray(point_inds)
        else:
            point_inds = data['tree'].query(center_point, k=num_points)[1][0]
        random.shuffle(point_inds)

        return point_inds

class spatially_regular_sampler(object):
    """Spatially regularSampler sampler for semantic segmentation datasets."""

    def __init__(self, data_split):
        self.data_split = data_split
        self.length = len(data_split)
        self.split = self.data_split.split

    def __len__(self):
        return self.length

    def initialize_with_dataset(self, dataset): # train_dataset, valid_dataset, test_dataset
        self.min_possibilities = []
        self.possibilities = []

        self.length = len(dataset)
        data_split = self.data_split

        for index in range(len(data_split)):
            attr = data_split.get_attr(index)
            if dataset.cache_convert:
                data = dataset.cache_convert(attr['name'])
            elif dataset.preprocess:
                data = dataset.preprocess(data_split.get_data(index), attr)
            else:
                data = data_split.get_data(index)

            pc = data['points']
            self.possibilities += [np.random.rand(pc.shape[0]) * 1e-3]
            self.min_possibilities += [float(np.min(self.possibilities[-1]))]

    def get_cloud_sampler(self):

        def gen_train():
            for i in range(self.length):
                self.cloud_id = int(np.argmin(self.min_possibilities))
                yield self.cloud_id

        def gen_test():
            curr_could_id = 0
            while curr_could_id < self.length:
                if self.min_possibilities[curr_could_id] > 0.5:
                    curr_could_id = curr_could_id + 1
                    continue
                self.cloud_id = curr_could_id

                yield self.cloud_id

        if self.split in ['train', 'validation', 'valid', 'training']:
            gen = gen_train
        else:
            gen = gen_test
        return gen()

    def get_sampled_points(self, data, cloud_id):
        # cfg = self.data_split.cfg
        # noise_init=cfg.get_input.parameter['noise_init']
        # # cloud_id = int(np.argmin(self.min_possibilities))
        # # if self.split == 'test':
        # #     cloud_id = self.cloud_id
        # point_ind = np.argmin(self.possibilities[cloud_id])
        # center_point = data['points'][point_ind, :].reshape(1,-1)
        # noise = np.random.normal(scale=noise_init / 10, size=center_point.shape)
        # center_point = center_point + noise.astype(center_point.dtype)      # 论文代码加入了噪声

        # if 'num_points' in cfg.get_input.parameter.keys():
        #     num_points = cfg.get_input.parameter['num_points']
        #     if (data['points'].shape[0] < num_points):                       # open3d-ml中随机重复选点补充到num_point
        #         diff = num_points - data['points'].shape[0]
        #         point_inds = np.array(range(data['points'].shape[0]))
        #         point_inds = list(point_inds) + list(random.choices(point_inds, k=diff))
        #         point_inds = np.asarray(point_inds)
        #         # point_inds = data['tree'].query(center_point, k=data['points'].shape[0])[1][0] # 论文只选data['points'].shape[0]个点
        #     else:
        #         point_inds = data['tree'].query(center_point, k=num_points)[1][0]

        # elif 'radius' in cfg.get_input.parameter.keys():
        #     radius = cfg.get_input.parameter['radius']
        #     point_inds = data['tree'].query_radius(center_point, r=radius)[0]
            
        #     random.shuffle(point_inds) # 方便随机采样
        #     input_points = data['points'][point_inds] 
        #     dists = np.sum(np.square((input_points - center_point).astype(np.float32)), axis=1)
        #     delta = np.square(1 - dists / np.max(dists))
        #     self.possibilities[cloud_id][point_inds] += delta
        #     self.min_possibilities[cloud_id] = float(np.min(self.possibilities[cloud_id]))
        
        cfg = self.data_split.cfg
        noise_init=cfg.get_input.parameter['noise_init']
        search_tree = data['tree']
        pc = data['points']
        n = 0
        while n < 2:
            center_id = np.argmin(self.possibilities[cloud_id])
            center_point = pc[center_id, :].reshape(1, -1)
            # 论文代码加入了噪声
            noise = np.random.normal(scale=noise_init / 10, size=center_point.shape)
            center_point = center_point + noise.astype(center_point.dtype)      
            if 'radius' in cfg.get_input.parameter.keys():
                radius = cfg.get_input.parameter['radius']
                point_inds = search_tree.query_radius(center_point, r=radius)[0]
            elif 'num_points' in cfg.get_input.parameter.keys():
                num_points = cfg.get_input.parameter['num_points']
                if (pc.shape[0] < num_points):
                    diff = num_points - pc.shape[0]
                    point_inds = np.array(range(pc.shape[0]))
                    point_inds = list(point_inds) + list(random.choices(point_inds, k=diff))
                    point_inds = np.asarray(point_inds)
                else:
                    point_inds = search_tree.query(center_point, k=num_points)[1][0]
            n = len(point_inds)
            if n < 2:
                self.possibilities[cloud_id][center_id] += 0.001

        random.shuffle(point_inds)
        pc = pc[point_inds]
        dists = np.sum(np.square((pc - center_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibilities[cloud_id][point_inds] += delta
        new_min = float(np.min(self.possibilities[cloud_id]))
        self.min_possibilities[cloud_id] = new_min

        return point_inds
