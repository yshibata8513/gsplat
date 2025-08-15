#!/usr/bin/env python3
"""
PandaSet抽出データの確認用可視化プロットを生成するツール

各カメラ画像について以下の5つの画像を生成：
1. オリジナルカメラ画像
2. 静的点群をカメラに投影
3. 静的点群の投影（黒背景）
4. 動的物体と統合した点群の投影（カメラ画像背景）
5. 動的物体と統合した点群の投影（黒背景）
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import cv2

# PLY読み込み用
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Warning: plyfile not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement

# neurad-studioのパスを追加
sys.path.append("/gs_workspace/neurad-studio")

from nerfstudio.cameras.lidars import transform_points_pairwise


class PandaSetVisualizer:
    def _dbg_report_depth_stats(self, uvz):
        if uvz.size == 0:
            print('[dbg] no projected points')
            return
        z = uvz[:,2]
        print(f"[dbg] depth>0 ratio: {(z>0).mean():.3f}, min={z.min():.3f}, max={z.max():.3f}")

    """PandaSet抽出データの可視化クラス"""
    
    def __init__(self, extracted_data_dir: Path, pandaset_path: Path, sequence: str):
        self.extracted_data_dir = Path(extracted_data_dir)
        self.pandaset_path = Path(pandaset_path)
        self.sequence = sequence
        
        # 抽出データを読み込み
        self._load_extracted_data()
        
        print(f"Loaded visualization data:")
        print(f"  Static points: {self.static_points.shape[0]}")
        print(f"  Dynamic actors: {len(self.dynamic_points)}")
        print(f"  Camera frames: {len(self.camera_data['frames'])}")
    
    def _load_extracted_data(self):
        """抽出されたデータを読み込み"""
        # 静的点群
        static_ply = PlyData.read(str(self.extracted_data_dir / "static_points.ply"))
        static_vertices = static_ply['vertex']
        self.static_points = np.column_stack([
            static_vertices['x'], static_vertices['y'], static_vertices['z'],
            static_vertices['red'], static_vertices['green'], static_vertices['blue']
        ])
        
        # 動的物体点群
        self.dynamic_points = {}
        dynamic_dir = self.extracted_data_dir / "dynamic_points"
        for ply_file in dynamic_dir.glob("actor_*.ply"):
            actor_id = int(ply_file.stem.split('_')[1])
            dynamic_ply = PlyData.read(str(ply_file))
            dynamic_vertices = dynamic_ply['vertex']
            self.dynamic_points[actor_id] = np.column_stack([
                dynamic_vertices['x'], dynamic_vertices['y'], dynamic_vertices['z'],
                dynamic_vertices['red'], dynamic_vertices['green'], dynamic_vertices['blue']
            ])
        
        # アクターポーズ
        with open(self.extracted_data_dir / "actor_poses.json") as f:
            self.actor_poses = json.load(f)
        
        # カメラデータ
        with open(self.extracted_data_dir / "camera_data.json") as f:
            self.camera_data = json.load(f)
        
        # メタデータ
        with open(self.extracted_data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
    
    def _load_camera_image(self, frame_id: int) -> Optional[np.ndarray]:
        """カメラ画像を読み込み（extracted_dataから）"""
        camera_frame = self.camera_data["frames"][frame_id]
        
        # extracted_dataディレクトリ内でカメラ画像を探す
        camera_images_dir = self.extracted_data_dir / "camera_images"
        
        if camera_images_dir.exists():
            # カメラ画像が保存されている場合
            image_path = camera_images_dir / f"frame_{frame_id:03d}.jpg"
            if image_path.exists():
                image = cv2.imread(str(image_path))
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # カメラ画像が見つからない場合は、camera_dataの情報から合成画像を生成
        height, width = int(camera_frame["height"]), int(camera_frame["width"])
        
        # グレーのグラデーション背景を生成（デバッグ用）
        image = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # フレーム情報を描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {frame_id}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        return image
    
    def _transform_dynamic_points_to_world(self, actor_id: int, time: float) -> np.ndarray:
        """動的物体の点群を指定時刻のワールド座標に変換"""
        if actor_id not in self.dynamic_points:
            return np.empty((0, 6))
        
        if str(actor_id) not in self.actor_poses:
            return self.dynamic_points[actor_id]  # ポーズ情報がない場合はそのまま
        
        actor_pose_data = self.actor_poses[str(actor_id)]
        poses = actor_pose_data["poses"]
        
        if not poses:
            return self.dynamic_points[actor_id]
        
        # 指定時刻に最も近いポーズを見つける
        closest_pose = min(poses, key=lambda p: abs(p["time"] - time))
        pose_matrix = np.array(closest_pose["pose"])  # 4x4 matrix
        
        # 【座標変換3】Actor Local座標 → World座標（可視化時）
        # local_points[:, :3]: Actor local coordinates [N, 3] (from 座標変換2で保存)
        # pose_matrix: Actor-to-World transformation matrix [4, 4] (時刻time時点)
        # points_homogeneous: Homogeneous coordinates [N, 4]
        local_points = self.dynamic_points[actor_id].copy()
        points_xyz = local_points[:, :3]  # Actor local coordinates
        colors = local_points[:, 3:]
        
        # 同次座標に変換してワールド座標へ
        points_homogeneous = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        world_points_homogeneous = points_homogeneous @ pose_matrix.T  # Local→World transformation
        world_points = world_points_homogeneous[:, :3]  # World coordinates
        
        return np.hstack([world_points, colors])
    
    def _depth_to_color(self, depths: np.ndarray) -> np.ndarray:
        """深度値をカラーマップで色分け（jet colormap使用）"""
        import matplotlib.pyplot as plt
        
        # 深度の正規化（1m-50m程度の範囲を想定）
        min_depth = max(1.0, depths.min())
        max_depth = min(100.0, depths.max())
        
        # 正規化した深度値（0-1の範囲）
        normalized_depths = np.clip((depths - min_depth) / (max_depth - min_depth), 0, 1)
        
        # jetカラーマップで色分け
        cmap = plt.cm.jet
        colors = cmap(normalized_depths)[:, :3]  # RGBのみ取得（alphaは除く）
        
        # 0-255スケールに変換
        return (colors * 255).astype(np.uint8)
    
    def _project_points_to_camera(self, points_3d: np.ndarray, camera_frame: Dict, 
                                 use_depth_coloring: bool = False, is_dynamic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """3D点をカメラ平面に投影（NerfstudioのOpenGL座標→OpenCV座標へ変換してから射影）"""
        # カメラの内部パラメータ（ピクセル単位）
        K = np.array([
            [camera_frame["fx"], 0, camera_frame["cx"]],
            [0, camera_frame["fy"], camera_frame["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # 【座標変換4】World座標 → Camera座標（NeuRAD-studio OpenGL→OpenCV変換）
        # points_xyz: World coordinates [N, 3] (from 座標変換1 or 座標変換3)
        # camera_to_world: Camera-to-World matrix [3,4] or [4,4] (OpenGL座標系)
        # T_w_cam: World-to-Camera matrix (OpenGL)
        # M_gl2cv: OpenGL→OpenCV coordinate conversion matrix
        # T_cam_w_cv: World-to-Camera matrix (OpenCV座標系)
        camera_to_world = np.array(camera_frame["camera_to_world"], dtype=np.float32)

        # 3x4を4x4に変換
        if camera_to_world.shape == (3, 4):
            T_w_cam = np.eye(4, dtype=np.float32)
            T_w_cam[:3, :] = camera_to_world
        else:
            T_w_cam = camera_to_world

        # World→Camera(OpenGL) へ
        T_cam_w_gl = np.linalg.inv(T_w_cam)

        # OpenGLカメラ座標 → OpenCVカメラ座標 への変換（x:同じ, y/z:反転）
        M_gl2cv = np.eye(4, dtype=np.float32)
        M_gl2cv[1, 1] = -1.0  # y軸反転
        M_gl2cv[2, 2] = -1.0  # z軸反転

        # World→Camera(OpenCV)
        T_cam_w_cv = M_gl2cv @ T_cam_w_gl

        # 点群をワールド座標からカメラ(OpenCV)座標へ変換
        points_xyz = points_3d[:, :3].astype(np.float32)  # World coordinates
        colors = points_3d[:, 3:]

        xyz_homogeneous = np.c_[points_xyz, np.ones(len(points_xyz), dtype=np.float32)]
        xyz_cam = (T_cam_w_cv @ xyz_homogeneous.T).T[:, :3]  # World→Camera(OpenCV)

        # カメラ前方の点のみ選択（>0m）
        front_mask = xyz_cam[:, 2] > 0.0
        xyz_cam = xyz_cam[front_mask]
        colors = colors[front_mask]

        if len(xyz_cam) == 0:
            return np.empty((0, 2)), np.empty((0, 3))

        # カメラ座標から画像座標へ投影
        uv = (K @ xyz_cam.T).T
        uv = uv / uv[:, 2:3]  # 透視除算

        # 画像範囲内の点のみ選択
        width, height = int(camera_frame["width"]), int(camera_frame["height"])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        
        # 深度・動的物体に基づく色付け
        final_colors = colors[inside]
        if use_depth_coloring and not is_dynamic:
            # 静的点群：jetカラーマップで深度色分け
            depths = xyz_cam[inside, 2]  # Z座標（深度）
            final_colors = self._depth_to_color(depths)
            
        elif is_dynamic:
            # 動的点群：白色
            final_colors = np.full((len(final_colors), 3), 255, dtype=np.uint8)

        return uv[inside, :2], final_colors
    def _render_points_on_image(self, image: np.ndarray, projected_points: np.ndarray, 
                               colors: np.ndarray, point_size: int = 2) -> np.ndarray:
        """画像上に点群を描画"""
        result_image = image.copy()
        
        for (x, y), color in zip(projected_points.astype(int), colors.astype(int)):
            cv2.circle(result_image, (x, y), point_size, color[:3].tolist(), -1)
        
        return result_image
    
    def _create_point_cloud_image(self, projected_points: np.ndarray, colors: np.ndarray, 
                                 width: int, height: int, point_size: int = 2) -> np.ndarray:
        """点群のみの画像を作成（黒背景）"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for (x, y), color in zip(projected_points.astype(int), colors.astype(int)):
            cv2.circle(image, (x, y), point_size, color[:3].tolist(), -1)
        
        return image
    
    def create_visualization_for_frame(self, frame_id: int, output_dir: Path):
        """指定フレームの5つの可視化画像を生成"""
        camera_frame = self.camera_data["frames"][frame_id]
        time = camera_frame["time"]
        width, height = int(camera_frame["width"]), int(camera_frame["height"])
        
        # 1. オリジナルカメラ画像
        original_image = self._load_camera_image(frame_id)
        
        # 2. 静的点群の投影（深度で色分け）
        static_points_2d, static_colors = self._project_points_to_camera(
            self.static_points, camera_frame, use_depth_coloring=True, is_dynamic=False
        )
        static_on_image = self._render_points_on_image(original_image, static_points_2d, static_colors)
        
        # 3. 静的点群のみ（黒背景）
        static_only_image = self._create_point_cloud_image(static_points_2d, static_colors, width, height)
        
        # 4. 統合点群（静的+動的）の投影
        # 静的点群（深度色分け）+ 動的点群（白色）
        combined_points_list = []
        combined_colors_list = []
        
        if len(static_points_2d) > 0:
            combined_points_list.append(static_points_2d)
            combined_colors_list.append(static_colors)
        
        # 動的点群（白色）
        for actor_id in self.dynamic_points.keys():
            actor_world_points = self._transform_dynamic_points_to_world(actor_id, time)
            if len(actor_world_points) > 0:
                dynamic_points_2d, dynamic_colors = self._project_points_to_camera(
                    actor_world_points, camera_frame, use_depth_coloring=False, is_dynamic=True
                )
                if len(dynamic_points_2d) > 0:
                    combined_points_list.append(dynamic_points_2d)
                    combined_colors_list.append(dynamic_colors)
        
        if combined_points_list:
            combined_points_2d = np.vstack(combined_points_list)
            combined_colors = np.vstack(combined_colors_list)
            combined_on_image = self._render_points_on_image(original_image, combined_points_2d, combined_colors)
        else:
            combined_on_image = original_image.copy()
        
        # 5. 統合点群のみ（黒背景）
        combined_only_image = self._create_point_cloud_image(combined_points_2d, combined_colors, width, height)
        
        # 画像を保存
        frame_output_dir = output_dir / f"frame_{frame_id:03d}"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(frame_output_dir / "1_original.jpg"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_output_dir / "2_static_on_image.jpg"), cv2.cvtColor(static_on_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_output_dir / "3_static_only.jpg"), cv2.cvtColor(static_only_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_output_dir / "4_combined_on_image.jpg"), cv2.cvtColor(combined_on_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_output_dir / "5_combined_only.jpg"), cv2.cvtColor(combined_only_image, cv2.COLOR_RGB2BGR))
        
        print(f"Generated visualization for frame {frame_id} -> {frame_output_dir}")
    
    def create_sample_visualizations(self, num_frames: int = 5, output_dir: Path = None):
        """全カメラから均等に選んだフレームの可視化を生成"""
        if output_dir is None:
            output_dir = Path("./visualization_output")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # PandaSetの6カメラ構成を想定（240フレーム = 6カメラ×40フレーム）
        num_available_frames = len(self.camera_data["frames"])
        num_cameras = 6
        frames_per_camera = num_available_frames // num_cameras
        
        selected_frames = []
        frames_per_camera_to_select = max(1, num_frames // num_cameras)
        
        # 各カメラから均等にフレームを選択
        for camera_id in range(num_cameras):
            camera_start = camera_id * frames_per_camera
            camera_end = (camera_id + 1) * frames_per_camera
            
            # 各カメラから指定数のフレームをランダム選択
            camera_frames = list(range(camera_start, camera_end))
            selected_camera_frames = random.sample(camera_frames, 
                                                 min(frames_per_camera_to_select, len(camera_frames)))
            selected_frames.extend(selected_camera_frames)
        
        # 不足分があれば追加でランダム選択
        if len(selected_frames) < num_frames:
            remaining = num_frames - len(selected_frames)
            all_frames = set(range(num_available_frames))
            available_frames = list(all_frames - set(selected_frames))
            additional_frames = random.sample(available_frames, min(remaining, len(available_frames)))
            selected_frames.extend(additional_frames)
        
        selected_frames.sort()
        
        print(f"Creating visualizations for {len(selected_frames)} frames from all cameras: {selected_frames}")
        
        for frame_id in selected_frames:
            self.create_visualization_for_frame(frame_id, output_dir)
        
        print(f"\nVisualization complete! Check {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualization plots for extracted PandaSet data")
    parser.add_argument("--extracted-data-dir", type=str, default="./extracted_data", 
                       help="Directory containing extracted data")
    parser.add_argument("--pandaset-path", type=str, default="./pandaset", 
                       help="Path to PandaSet dataset")
    parser.add_argument("--sequence", type=str, default="001", help="Sequence name")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to visualize")
    parser.add_argument("--output-dir", type=str, default="./visualization_output", help="Output directory")
    
    args = parser.parse_args()
    
    visualizer = PandaSetVisualizer(
        extracted_data_dir=Path(args.extracted_data_dir),
        pandaset_path=Path(args.pandaset_path),
        sequence=args.sequence
    )
    
    visualizer.create_sample_visualizations(
        num_frames=args.num_frames,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()