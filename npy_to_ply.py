import open3d as o3d
import numpy as np
from tqdm import tqdm

def write_result(filename, xyzs, rgbs, gt_labels, **kwargs):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property float gt_label\n')
        for key in kwargs.keys():
            f.write(f'property float {str(key)}\n')
        f.write('end_header\n')
        t = tqdm(range(len(gt_labels)))
        for i in t:
            x, y, z = xyzs[i]
            r, g, b = rgbs[i]
            gt_label = gt_labels[i]
            f.write(f'{x} {y} {z} {r} {g} {b} {gt_label}')
            for value in kwargs.values():
                pred_label = value[i]
                f.write(f' {pred_label}')
            f.write('\n')
        t.close()
def write_predLabel(xyzs, rgbs, scalar_Labels, pred_Labels0, pred_Labels1, pred_Labels2, fileName='pred_L002.ply'):
  
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property float scalar_Label\n')
        f.write('property float pred_Label0\n')
        f.write('property float pred_Label1\n')
        f.write('property float pred_Label2\n')
        f.write('end_header\n')
        t = tqdm(range(len(xyzs)))
        for i in t:
            r, g, b = rgbs[i]
            x, y, z = xyzs[i]
            scalar_Label = scalar_Labels[i]
            pred_Label0 = pred_Labels0[i]
            pred_Label1 = pred_Labels1[i]
            pred_Label2 = pred_Labels2[i]
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(x, y, z, r, g, b, scalar_Label, 
                                                        pred_Label0, pred_Label1, pred_Label2))
        t.close()

if __name__ == "__main__":

    ori_ply = o3d.t.io.read_point_cloud('original_data/Toronto_3D/L002.ply')
    xyzs = (ori_ply.point["positions"]).numpy()
    rgbs = (ori_ply.point["colors"]).numpy()
    gt_labels = (ori_ply.point["scalar_Label"]).numpy().astype(np.int32).reshape((-1,))
    RandLANet_xyz = np.load('all_results/RandLA-Net_Toronto-3D_xyz_wce/predict_results/last_epoch/L002.npy').reshape((-1,))
    Ours_xyz = np.load('all_results/RandLA-Net2_Toronto-3D_xyz/class_balanced_sampler/predict_results/train_best/L002.npy').reshape((-1,))
    RandLANet_rgb = np.load('all_results/RandLA-Net_Toronto-3D_xyz+rgb_wce/predict_results/last_epoch/L002.npy').reshape((-1,))
    Ours_rgb = np.load('all_results/RandLANet2_Toronto-3D_xyz+rgb/class_balanced_sampler/focal/predict_results/last_epoch/L002.npy').reshape((-1,))

    write_result( 
                    './pred_L002.ply',
                    xyzs, 
                    rgbs, 
                    gt_labels, 
                    RandLANet_xyz=RandLANet_xyz, 
                    Ours_xyz=Ours_xyz,
                    RandLANet_rgb=RandLANet_rgb,
                    Ours_rgb=Ours_rgb
    )
   