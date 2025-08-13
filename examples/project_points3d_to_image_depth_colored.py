#!/usr/bin/env python3
"""
Project COLMAP points3D.bin to images with depth-based color coding

This script visualizes 3D point clouds from COLMAP reconstructions by projecting them
onto 2D images with depth-based color coding. It uses gsplat's coordinate system and
transformation pipeline for accurate projection.

Features:
- Projects all COLMAP points3D.bin points to camera images
- Color codes points by depth (near=purple, far=yellow by default)
- Creates both overlay (points on original image) and black background visualizations
- Supports multiple datasets (bicycle, facade, etc.)
- Configurable colormaps (viridis, plasma, jet, etc.)

Usage:
    python project_points3d_to_image_depth_colored.py [dataset] [num_images] [colormap]
    
Examples:
    python project_points3d_to_image_depth_colored.py bicycle 3 viridis
    python project_points3d_to_image_depth_colored.py facade 5 plasma
    python project_points3d_to_image_depth_colored.py  # defaults to bicycle
"""

import os
import sys
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from datasets.colmap import Parser

def depth_to_color(depths, colormap='viridis'):
    """
    Convert depth values to RGB colors using matplotlib colormap
    """
    # Normalize depths to 0-1 range
    depths_normalized = (depths - depths.min()) / (depths.max() - depths.min())

    # depths_normalized = 1. - depths_normalized
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors = cmap(depths_normalized)
    
    # Convert to BGR for OpenCV (matplotlib returns RGB)
    colors_bgr = (colors[:, [2, 1, 0]] * 255).astype(np.uint8)
    
    return colors_bgr

def project_colmap_points_with_depth(parser, image_index):
    """
    Project COLMAP points3D.bin points with depth information
    """
    # Get camera parameters
    camtoworld = parser.camtoworlds[image_index]
    camera_id = parser.camera_ids[image_index] 
    K = parser.Ks_dict[camera_id]
    width, height = parser.imsize_dict[camera_id]
    
    # Use all COLMAP points
    points_world = parser.points
    
    # Project using gsplat method
    worldtocam = np.linalg.inv(camtoworld)
    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    depths = points_cam[:, 2]
    
    # Visibility filtering
    valid = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
             (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) &
             (depths > 0))
    
    return points_2d[valid], depths[valid], np.where(valid)[0]

def visualize_depth_colored(image_path, points_2d, depths, output_path, mode='overlay', colormap='viridis'):
    """Create depth-colored visualization of projected points."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if mode == 'black_background':
        image = np.zeros_like(image)
    
    # Get colors based on depth
    colors = depth_to_color(depths, colormap)
    
    # Draw points with depth colors
    for i, (x, y) in enumerate(points_2d):
        color = tuple(int(c) for c in colors[i])
        cv2.circle(image, (int(x), int(y)), 2, color, -1)
    
    cv2.imwrite(output_path, image)
    print(f"Saved depth-colored visualization: {output_path}")

def create_colorbar(depths, output_path, colormap='viridis'):
    """Create a colorbar showing depth mapping"""
    fig, ax = plt.subplots(figsize=(8, 1))
    
    # Create colorbar
    cmap = cm.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=depths.min(), vmax=depths.max()))
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label('Depth', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Parse command line arguments
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'bicycle'
    num_images_to_process = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    colormap = sys.argv[3] if len(sys.argv) > 3 else 'viridis'
    
    # Dataset configuration
    dataset_configs = {
        'bicycle': {
            'data_dir': "/workspace/gsplat/examples/data/360_v2/bicycle",
            'output_dir': "/workspace/gsplat/examples/bicycle_depth_colored",
            'default_num_images': 3
        },
        'facade': {
            'data_dir': "/workspace/gsplat/examples/data/facade",
            'output_dir': "/workspace/gsplat/examples/facade_depth_colored",
            'default_num_images': 5
        }
    }
    
    if dataset not in dataset_configs:
        print(f"Error: Unknown dataset '{dataset}'. Available: {list(dataset_configs.keys())}")
        sys.exit(1)
    
    config = dataset_configs[dataset]
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    
    # Use default number of images for dataset if not specified
    if len(sys.argv) <= 2:
        num_images_to_process = config['default_num_images']
    
    print(f"Dataset: {dataset}")
    print(f"Processing {num_images_to_process} images")
    print(f"Colormap: {colormap}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Validate colormap
    available_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'turbo', 'cool', 'hot']
    if colormap not in available_colormaps:
        print(f"Warning: '{colormap}' may not be available. Recommended: {available_colormaps}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize parser
    print("Loading COLMAP data...")
    parser = Parser(
        data_dir=data_dir,
        factor=4,
        normalize=True,
        test_every=8
    )
    
    print(f"Loaded {len(parser.image_names)} images")
    print(f"Total points3D: {len(parser.points)}")
    
    # Process images
    results = []
    all_depths = []
    
    for i in range(min(num_images_to_process, len(parser.image_names))):
        image_name = parser.image_names[i]
        image_path = parser.image_paths[i]
        
        print(f"\nProcessing image {i+1}: {image_name}")
        
        # Project all COLMAP points to this image
        points_2d, depths, valid_indices = project_colmap_points_with_depth(parser, i)
        all_depths.extend(depths)
        
        total_points = len(parser.points)
        projected_points = len(points_2d)
        percentage = projected_points / total_points * 100
        
        print(f"  Projected {projected_points}/{total_points} points ({percentage:.1f}%)")
        print(f"  Depth range: {depths.min():.3f} - {depths.max():.3f}")
        
        # Create depth-colored visualizations (clean filename)
        clean_image_name = os.path.basename(image_name)
        overlay_path = os.path.join(output_dir, f"depth_{colormap}_{i:03d}_{clean_image_name}_overlay.jpg")
        black_bg_path = os.path.join(output_dir, f"depth_{colormap}_{i:03d}_{clean_image_name}_black_bg.jpg")
        
        visualize_depth_colored(image_path, points_2d, depths, overlay_path, mode='overlay', colormap=colormap)
        visualize_depth_colored(image_path, points_2d, depths, black_bg_path, mode='black_background', colormap=colormap)
        
        # Store results
        results.append({
            "image_name": image_name,
            "projected_points": projected_points,
            "total_points": total_points,
            "percentage": percentage,
            "mean_depth": float(np.mean(depths)),
            "depth_range": [float(np.min(depths)), float(np.max(depths))]
        })
    
    # Create colorbar
    if all_depths:
        all_depths = np.array(all_depths)
        colorbar_path = os.path.join(output_dir, f"depth_colorbar_{colormap}.png")
        create_colorbar(all_depths, colorbar_path, colormap)
        print(f"Created colorbar: {colorbar_path}")
        print(f"Overall depth range: {all_depths.min():.3f} - {all_depths.max():.3f}")
    
    # Save summary
    summary = {
        "data_dir": data_dir,
        "total_colmap_points": len(parser.points),
        "colormap": colormap,
        "coordinate_system": "gsplat_normalized",
        "source": "points3D.bin_direct_depth_colored",
        "overall_depth_range": [float(all_depths.min()), float(all_depths.max())],
        "results": results
    }
    
    summary_path = os.path.join(output_dir, "depth_colored_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Results saved to: {output_dir}")
    print(f"Colormap used: {colormap}")
    print(f"Total COLMAP points: {len(parser.points)}")
    print(f"Images processed: {len(results)}")
    
    for result in results:
        print(f"  {result['image_name']}: {result['projected_points']}/{result['total_points']} "
              f"({result['percentage']:.1f}%) points visible")

if __name__ == "__main__":
    main()