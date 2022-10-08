import torch
import torch.nn as nn
import numpy as np

from datasets.utils.dataprocessing import DataProcessing as DP

__all__ = ['RandLANet']

class RandLANet(nn.Module):
    def __init__(self, cfg):            # 此处cfg为cfg.model
        self.cfg = cfg
        super().__init__()    
        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)
        # Encoder
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features       
        for i in range(len(cfg.sub_sampling_ratio)):
            self.encoder.append(LocalFeatureAggregation(dim_feature, cfg.dim_output[i], cfg.num_neighbours))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)
        self.encoder = nn.ModuleList(self.encoder)
        self.mlp = SharedMLP(dim_feature,
                             dim_feature,
                             activation_fn=nn.LeakyReLU(0.2))
        # Decoder
        self.decoder = []
        for i in range(len(cfg.sub_sampling_ratio)):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]
        self.decoder = nn.ModuleList(self.decoder)
        self.fc1 = nn.Sequential(SharedMLP(dim_feature, 64, activation_fn=nn.LeakyReLU(0.2)),
                                 SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)), nn.Dropout(0.5),
                                 SharedMLP(32, cfg.num_classes, bn=False))

    def transform(self, data):
        pc = data['points']
        input_points = []
        input_pools_id = []
        input_neighbours_id = []
        input_up_samples_id = []

        add_fps = self.cfg.add_fps 
        sub_sampling_ratio = self.cfg.sub_sampling_ratio
        num_layers = len(sub_sampling_ratio)
        num_neighbours = self.cfg.num_neighbours
        for i in range(num_layers):
            neighbour_idx = DP.knn_search(pc, pc, num_neighbours)
            if add_fps: # 在Toronto3D上没有效果
                if i<3:
                    sub_points = pc[:pc.shape[0] // sub_sampling_ratio[i], :]
                    pool_i = neighbour_idx[:pc.shape[0] // sub_sampling_ratio[i], :]
                else:
                    sub_points_index = DP.farthest_point_sample(pc,pc.shape[0] // sub_sampling_ratio[i])
                    sub_points, pool_i = pc[sub_points_index], neighbour_idx[sub_points_index]
            else:
                sub_points = pc[:pc.shape[0] // sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:pc.shape[0] // sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbours_id.append(neighbour_idx.astype(np.int64))
            input_pools_id.append(pool_i.astype(np.int64))
            input_up_samples_id.append(up_i.astype(np.int64))
            pc = sub_points
        
        data['points'] = input_points 
        data['neighbor_indices'] = input_neighbours_id
        data['sub_idx'] = input_pools_id
        data['interp_idx'] = input_up_samples_id
        if data['point_inds'] is None:
            del data['point_inds']
        if data['proj_inds'] is None:
            del data['proj_inds']

        return data
    
    def forward(self, data):
        cfg = self.cfg
        feat = data['features']
        coords_list = data['points']
        neighbor_indices_list = data['neighbor_indices']
        subsample_indices_list = data['sub_idx']
        interpolation_indices_list = data['interp_idx']
    
        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)  # (B, dim_feature, N, 1)
        feat = self.bn0(feat)  # (B, d, N, 1)
        l_relu = nn.LeakyReLU(0.2)
        feat = l_relu(feat)
        encoder_feat_list = []
        for i in range(len(cfg.sub_sampling_ratio)):
            feat_encoder_i = self.encoder[i](coords_list[i], feat, neighbor_indices_list[i]) # feat_encoder_i: (B,dim,N,1)
            feat_sampled_i = self.get_sampled(feat_encoder_i, subsample_indices_list[i])   # subsample_indices_list[i]: (B,N',16) N' --> N//sub_sample_ratio[i] feat_sampler_i : (B,dim,N',1)
            if i == 0:
                encoder_feat_list.append(feat_encoder_i.clone())
            encoder_feat_list.append(feat_sampled_i.clone())
            feat = feat_sampled_i
        feat = self.mlp(feat)
        for i in range(len(cfg.sub_sampling_ratio)):
            feat_interpolation_i = self.nearest_interpolation(feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = torch.cat([encoder_feat_list[-i - 2], feat_interpolation_i], dim=1)
            feat_decoder_i = self.decoder[i](feat_decoder_i)
            feat = feat_decoder_i
        scores = self.fc1(feat)
        return scores.squeeze(3).transpose(1, 2)  

    @staticmethod
    def get_sampled(feature, pool_idx):  # (KNN) gather + maxpool
        B, d, N, K = feature.size()
        B, N_, K = pool_idx.size()
        feature = feature.expand(B, d, N, K)
        pool_idx = pool_idx.unsqueeze(1).expand(B, d, N_, K)
        pool_features = torch.gather(feature,2,pool_idx)
        pool_features, _ = torch.max(pool_features, -1, keepdim=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(3)
        d = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]
        interp_idx = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx = interp_idx.unsqueeze(1).expand(batch_size, d, -1)
        interpolatedim_features = torch.gather(feature, 2, interp_idx)
        interpolatedim_features = interpolatedim_features.unsqueeze(3) 
        return interpolatedim_features

class SharedMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 bn=True,
                 activation_fn=None):
        super(SharedMLP, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(kernel_size - 1) // 2)
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalSpatialEncoding(nn.Module):
    def __init__(self, dim_in, dim_out, num_neighbours, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()
        self.num_neighbours = num_neighbours
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbor(self, coords, neighbor_indices):
        B, N, K = neighbor_indices.size()
        dim = coords.shape[2]
        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, dim, N, K)
        neighbor_coords = torch.gather(extended_coords, 2, extended_indices)  # (B, dim, N, K)
        return neighbor_coords

    def forward(self, coords, features, neighbor_indices, relative_features=None):
        B, N, K = neighbor_indices.size()
        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(torch.sum(torch.square(relative_pos), dim=1, keepdim=True))
            relative_features = torch.cat([relative_dist, relative_pos, extended_coords
                                            ,neighbor_coords
                                            ],dim=1)
        else:
            if relative_features is None:
                raise ValueError("LocalSpatialEncoding: Require relative_features for second pass.")
        relative_features = self.mlp(relative_features)
        neighbor_features = self.gather_neighbor(features.transpose(1, 2).squeeze(3), neighbor_indices)
        return torch.cat([neighbor_features, relative_features], dim=1), relative_features

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels, out_channels, activation_fn=nn.LeakyReLU(0.2))
    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)

class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbours):
        super(LocalFeatureAggregation, self).__init__()
        self.num_neighbours = num_neighbours
        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, num_neighbours, encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbours)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        x = self.mlp1(feat)
        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)
        x, _ = self.lse2(coords, x, neighbor_indices, relative_features=neighbor_features)
        x = self.pool2(x)
        return self.lrelu(self.mlp2(x) + self.shortcut(feat))
