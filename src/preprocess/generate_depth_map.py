import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np
#import scipy.misc as ssc
import imageio



def generate_dispariy_from_velo(pc_velo, height, width, calib):
    ##Projection from lidar to camera 2: proj_velo2cam2 = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref 
    pts_2d = calib.project_velo_to_image(pc_velo)#input: (96572, 3), output: (96572, 2)
    #step1: project_velo_to_ref: np.dot(pts_3d_velo(nx4), np.transpose(self.V2C)), euclidean transformation from lidar to reference camera cam0
    #step2: project_ref_to_rect: np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref))), Rotation from reference camera coord to rect camera coord
    #step3: project_rect_to_image:  pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
    #pts_2d shape example: (96572, 2), 2: xy, 96572 number of points

    ## Filter lidar points to be within image FOV
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)#true false list
    imgfov_pc_velo = pc_velo[fov_inds, :]#only lidar points within FOV, sample shape: (8669, 3), 3:xyz
    imgfov_pts_2d = pts_2d[fov_inds, :]## Filter out pixels points
    #imgfov_pts_2d shape example: (8669, 2)

    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)#input: (8669, 3)
    #step1: pts_3d_ref = project_velo_to_ref(pts_3d_velo)
    #step2: project_ref_to_rect(pts_3d_ref)
    #imgfov_pc_rect contains depth, shape=(8669, 3), different from 3DVisualizeV2??

    depth_map = np.zeros((height, width)) - 1 #(2056, 2464)
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)#point xy become int index
    for i in range(imgfov_pts_2d.shape[0]):#8669
        depth = imgfov_pc_rect[i, 2] #3: xyz, 2 index means z depth, 17.04, 37.04
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    #parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    #parser.add_argument('--split_file', type=str, default='~/Kitti/object/train.txt')

    #parser.add_argument('--data_path', type=str, default='/Developer/Dataset/Argoverse/argoverse-conv-rect-mynew/training')
    parser.add_argument('--data_path', type=str, default='/Developer/Dataset/Argoverse/argoverse-conv-rect-all/training')
    parser.add_argument('--split_file', type=str, default='/Developer/3DObject/MyPseudoLidar/split/fullargo.txt')#all argo image files
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    depth_dir = args.data_path + '/depth_map/' 

    assert os.path.isdir(lidar_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(depth_dir): #created folder
        os.makedirs(depth_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    for fn in lidar_files:#1903 files
        predix = fn[:-4]#remove .bin, only leave name "000000"
        if predix not in file_names:
            continue
        calib_file = '{}{}.txt'.format(calib_dir, predix)#calib_dir already contains "/"
        calib = kitti_util.Calibration(calib_file)
        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3] #shape: (96572, 3)
        image_file = '{}{}.png'.format(image_dir, predix)
        #image = ssc.imread(image_file)
        image = imageio.imread(image_file)#(2056, 2464, 3), now: (514, 616, 3)
        height, width = image.shape[:2]
        depth_map = generate_dispariy_from_velo(lidar, height, width, calib)
        np.save(depth_dir + '/' + predix, depth_map)
        print('Finish Depth Map {}'.format(predix))
