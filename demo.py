import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import torch
import open3d as o3d

# Satisfy strict args for utils.arguments if missing
if '--dataset_root' not in sys.argv:
    sys.argv.extend(['--dataset_root', 'dummy_root'])
if '--camera' not in sys.argv:
    sys.argv.extend(['--camera', 'kinect'])

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp, pred_decode
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetector

def load_data(example_path, num_points=20000):
    color_path = os.path.join(example_path, 'color.png')
    depth_path = os.path.join(example_path, 'depth.png')
    meta_path = os.path.join(example_path, 'meta.mat')
    mask_path = os.path.join(example_path, 'workspace_mask.png')

    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"Color or Depth image not found in {example_path}")

    # Load images
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    
    # Load meta
    if os.path.exists(meta_path):
        meta = scio.loadmat(meta_path)
        if 'intrinsic_matrix' in meta and 'factor_depth' in meta:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        else:
            print("Warning: Meta file format unexpected, using defaults.")
            intrinsic = np.array([[631.5, 0, 638.5], [0, 631.2, 367.0], [0, 0, 1]])
            factor_depth = 1000.0
    else:
        print("Warning: meta.mat not found. Using default kinect intrinsics.")
        intrinsic = np.array([[631.5, 0, 638.5], [0, 631.2, 367.0], [0, 0, 1]]) 
        factor_depth = 1000.0

    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    # Create cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    # Masking
    depth_mask = (depth > 0)
    if os.path.exists(mask_path):
        workspace_mask = np.array(Image.open(mask_path)) > 0
        mask = depth_mask & workspace_mask
    else:
        mask = depth_mask

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) == 0:
        raise ValueError("No valid points found after masking!")

    # Sample points
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    ret_dict = {}
    ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
    ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
    ret_dict['coordinates_for_voxel'] = cloud_sampled.astype(np.float32) / cfgs.voxel_size
    
    return ret_dict

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    example_path = cfgs.example_path
    if not example_path:
         raise ValueError("Example path is not specified in arguments.")

    # Load Data
    print(f"Loading data from {example_path}...")
    data_dict = load_data(example_path, num_points=cfgs.num_point)
    
    # Prepare Batch
    batch_data = {}
    batch_data['point_clouds'] = torch.from_numpy(data_dict['point_clouds']).unsqueeze(0).to(device)
    # coordinates_for_voxel must be a list of arrays/tensors for the model's collation logic
    batch_data['coordinates_for_voxel'] = [torch.from_numpy(data_dict['coordinates_for_voxel']).to(device)]
    
    # Initialize Model
    print("Initializing model...")
    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    
    # Load Checkpoint
    if cfgs.checkpoint_path is None:
        raise ValueError("--checkpoint_path argument is required.")
        
    print(f"Loading checkpoint from {cfgs.checkpoint_path}...")
    if not os.path.isfile(cfgs.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {cfgs.checkpoint_path}")

    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
        
    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    print(f"Raw grasps found: {len(gg)}")

    # Collision Detection
    if cfgs.collision_thresh > 0:
        print("Running collision detection...")
        cloud = data_dict['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        print(f"Grasps after collision detection: {len(gg)}")


    # NMS
    print("Running NMS...")
    gg = gg.nms()
    gg = gg.sort_by_score() 
    gg = gg[:50] # Top 50, adjustable
    print(f"Final grasps to visualize: {len(gg)}")
    
    # Visualization
    print("Visualizing... (Close the window to exit)")
    grippers = gg.to_open3d_geometry_list()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_dict['point_clouds'])
    pcd.colors = o3d.utility.Vector3dVector(data_dict['cloud_colors'])
    
    o3d.visualization.draw_geometries([pcd, *grippers], window_name="EconomicGrasp Demo")

if __name__ == '__main__':
    main()
