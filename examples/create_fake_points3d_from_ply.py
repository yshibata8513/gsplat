#!/usr/bin/env python3
"""
Create a fake points3D.bin from PLY file for use with gsplat COLMAP parser

This script reads a PLY file (like merged_lidar_100k.ply) and creates minimal
COLMAP-compatible points3D.bin file that can be used with gsplat's Parser.

The fake points3D.bin contains:
- Point coordinates (X, Y, Z)
- RGB colors (if available in PLY, otherwise white)
- Minimal error values (set to 0.01)
- No track information (empty track lists)

Usage:
    python create_fake_points3d_from_ply.py [ply_file] [output_dir]
    
Example:
    python create_fake_points3d_from_ply.py data/facade/merged_lidar_100k.ply data/facade/sparse/0/
"""

import os
import sys
import struct
import numpy as np
import open3d as o3d
from pathlib import Path

def write_points3d_bin(output_path, points, colors, errors):
    """
    Write COLMAP points3D.bin file
    
    Format per point:
    - POINT3D_ID (8 bytes, uint64)
    - X, Y, Z (24 bytes, 3 × double)
    - R, G, B (3 bytes, 3 × uint8)
    - ERROR (8 bytes, double)
    - TRACK_LENGTH (8 bytes, uint64)
    - TRACK[] (TRACK_LENGTH × 8 bytes, pairs of IMAGE_ID, POINT2D_IDX)
    """
    
    num_points = len(points)
    print(f"Writing {num_points} points to {output_path}")
    
    with open(output_path, 'wb') as f:
        # Write number of points
        f.write(struct.pack('<Q', num_points))
        
        for i in range(num_points):
            point_id = i + 1  # COLMAP point IDs start from 1
            x, y, z = points[i]
            r, g, b = colors[i]
            error = errors[i]
            
            # Write point data
            f.write(struct.pack('<Q', point_id))  # POINT3D_ID
            f.write(struct.pack('<ddd', x, y, z))  # XYZ coordinates
            f.write(struct.pack('<BBB', r, g, b))  # RGB colors
            f.write(struct.pack('<d', error))      # ERROR
            f.write(struct.pack('<Q', 0))          # TRACK_LENGTH (empty track)
            # No track data since TRACK_LENGTH = 0
            
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}/{num_points} points")
    
    print(f"Successfully wrote points3D.bin with {num_points} points")

def load_ply_file(ply_path):
    """Load PLY file and extract points and colors"""
    print(f"Loading PLY file: {ply_path}")
    
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in PLY file: {ply_path}")
    
    # Extract points
    points = np.asarray(pcd.points, dtype=np.float64)
    
    # Extract colors
    if len(pcd.colors) > 0:
        colors_float = np.asarray(pcd.colors)  # Colors in [0,1] range
        colors = (colors_float * 255).astype(np.uint8)  # Convert to [0,255]
        print(f"Loaded {len(points)} points with colors")
    else:
        # Default to white if no colors
        colors = np.full((len(points), 3), 200, dtype=np.uint8)  # Light gray
        print(f"Loaded {len(points)} points without colors (using light gray)")
    
    # Create minimal errors (COLMAP reconstruction error)
    errors = np.full(len(points), 0.01, dtype=np.float64)  # Small constant error
    
    print(f"Point cloud statistics:")
    print(f"  Points: {len(points)}")
    print(f"  Bounding box: {points.min(axis=0)} to {points.max(axis=0)}")
    print(f"  Color range: {colors.min()}-{colors.max()}")
    print(f"  Error values: {errors.min()}-{errors.max()}")
    
    return points, colors, errors

def backup_original_points3d(sparse_dir):
    """Backup original points3D.bin to _points3D.bin if it exists"""
    points3d_path = sparse_dir / "points3D.bin"
    backup_path = sparse_dir / "_points3D.bin"
    
    if points3d_path.exists():
        if backup_path.exists():
            print(f"Backup already exists: {backup_path}")
        else:
            points3d_path.rename(backup_path)
            print(f"Backed up original points3D.bin to {backup_path}")
    else:
        print("No original points3D.bin found (no backup needed)")

def main():
    # Parse command line arguments
    ply_file = sys.argv[1] if len(sys.argv) > 1 else "data/facade/merged_lidar_100k.ply"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/facade/sparse/0"
    
    ply_path = Path(ply_file)
    sparse_dir = Path(output_dir)
    
    print(f"PLY file: {ply_path}")
    print(f"Output directory: {sparse_dir}")
    
    # Validate inputs
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)
    
    if not sparse_dir.exists():
        print(f"Creating output directory: {sparse_dir}")
        sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cameras.bin and images.bin exist (required for COLMAP sparse reconstruction)
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    
    if not cameras_bin.exists():
        print(f"Warning: {cameras_bin} not found. COLMAP parser may fail.")
    if not images_bin.exists():
        print(f"Warning: {images_bin} not found. COLMAP parser may fail.")
    
    # Backup original points3D.bin
    backup_original_points3d(sparse_dir)
    
    # Load PLY file
    try:
        points, colors, errors = load_ply_file(ply_path)
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        sys.exit(1)
    
    # Write fake points3D.bin
    output_path = sparse_dir / "points3D.bin"
    try:
        write_points3d_bin(output_path, points, colors, errors)
    except Exception as e:
        print(f"Error writing points3D.bin: {e}")
        sys.exit(1)
    
    print(f"\n=== SUCCESS ===")
    print(f"Created fake points3D.bin from PLY file")
    print(f"Input: {ply_path}")
    print(f"Output: {output_path}")
    print(f"Points: {len(points)}")
    print("\nYou can now use this with gsplat's COLMAP parser:")
    print(f"  python project_points3d_to_image_depth_colored.py facade")

if __name__ == "__main__":
    main()