import os
import cv2
import numpy as np
import tqdm

depth_scaling_factor = 5000.0 # 5000.0 for nyu (Microsoft Kinect)

def resize_depth_map(depth_map, width, height):
    return cv2.resize(depth_map, (width, height))

def calculate_depth_difference(gt_dir, estimated_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    common_files = set(os.listdir(gt_dir)).intersection(os.listdir(estimated_dir))
    diff_value = []
    gt_depth_mean = []
    estimated_depth_mean = []

    for filename in tqdm.tqdm(common_files, desc="Calculating Depth Differences"):

        gt_path = os.path.join(gt_dir, filename)
        estimated_path = os.path.join(estimated_dir, filename)

        gt_depth = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        estimated_depth = cv2.imread(estimated_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

        estimated_depth = resize_depth_map(estimated_depth, gt_depth.shape[1], gt_depth.shape[0])

        gt_depth = gt_depth / depth_scaling_factor
        estimated_depth = estimated_depth / depth_scaling_factor

        depth_difference = np.abs(gt_depth - estimated_depth)

        gt_depth_mean.append(np.mean(gt_depth))
        estimated_depth_mean.append(np.mean(estimated_depth))
        

        mean_abs_diff = np.mean(depth_difference)
        diff_value.append(mean_abs_diff)

        output_path = os.path.join(output_dir, f"{filename[:-4]}_difference.png")
        cv2.imwrite(output_path, (depth_difference * depth_scaling_factor).astype(np.uint16))
    
    print(f"Mean Absolute Difference: {np.mean(diff_value)}")
    

if __name__ == '__main__':
    gt_depth_directory = "data/slam/nyu/office_0008/depth"
    estimated_depth_directory = "nyu_depth_test"

    output_directory = "nyu_depth_diff"

    calculate_depth_difference(gt_depth_directory, estimated_depth_directory, output_directory)


