import os
import sys
import numpy as np
import pandas as pd
import cv2
import tqdm

depth_scaling_factor = 5000


def resize_depth_map(depth_map, width, height):
    return cv2.resize(depth_map, (width, height))


def calculate_depth_difference(gt_depth_map, estimated_depth_map) :

    estimated_depth_map = resize_depth_map(estimated_depth_map, gt_depth_map.shape[1], gt_depth_map.shape[0])

    gt_depth_map = gt_depth_map 
    estimated_depth_map = estimated_depth_map 

    depth_difference = np.abs(gt_depth_map - estimated_depth_map)

    return np.mean(depth_difference)


def sort_files(files) :
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        return files


def find_matching_dir_and_calc(gt_dir, estimated_dir) :

    depth_dirs = [d for d in os.listdir(estimated_dir) if os.path.isdir(os.path.join(estimated_dir, d))]
    depth_dirs = sorted(depth_dirs)

    for depth_dir in depth_dirs :

        gt_path = os.path.join(gt_dir, depth_dir+'.png')
        gt_depth = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

        files_dir = estimated_dir + '/' + depth_dir
        
        files = [f for f in os.listdir(files_dir) if f.endswith('.png')]

        diff_list = []
        
        files = sort_files(files)
        print(files)
        
        for file in files :

            estimated_path = files_dir + '/' + file
            estimated_depth = cv2.imread(estimated_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            
            diff = calculate_depth_difference(gt_depth, estimated_depth)

            diff_list.append(diff)

        print(f"{depth_dir} : {diff_list}")

if __name__ == "__main__" :

    depth_map_dir = "/home/eunjae/deepv2d/nyu_depth_test"
    gt_depth_dir = "/home/eunjae/deepv2d/data/slam/nyu/office_0008/depth"

    find_matching_dir_and_calc(gt_depth_dir, depth_map_dir)
        