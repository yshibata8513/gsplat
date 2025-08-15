#!/usr/bin/env python3
"""
PandaSet Data Extraction Tool

PandaSetデータセットから静的・動的点群とカメラデータを抽出するツール。
既存のSplatADコードを活用して実装。
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

# PLY書き込み用
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Warning: plyfile not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement

# neurad-studioのパスを追加
sys.path.append("/gs_workspace/neurad-studio")

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.data.utils.data_utils import points_in_box
from nerfstudio.model_components.dynamic_actors import DynamicActors, DynamicActorsConfig
from nerfstudio.utils.poses import inverse as pose_inverse


@dataclass
class ExtractionConfig:
    """データ抽出の設定"""
    pandaset_path: Path
    sequence: str
    static_points_count: int = 80000
    dynamic_points_per_actor: int = 5000
    output_dir: Path = Path("./extracted_data")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PandaSetDataExtractor:
    """PandaSetから静的・動的点群とカメラデータを抽出するクラス"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 出力ディレクトリの作成
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "dynamic_points").mkdir(exist_ok=True)
        
        # データ保存用
        self.static_points: Optional[Tensor] = None
        self.dynamic_points: Dict[int, Tensor] = {}
        self.actor_poses: Dict[int, Dict] = {}
        self.camera_data: Dict = {}
        self.metadata: Dict = {}
        
        print(f"Initializing PandaSet Data Extractor")
        print(f"  Dataset path: {config.pandaset_path}")
        print(f"  Sequence: {config.sequence}")
        print(f"  Device: {self.device}")
    
    def extract_data(self):
        """データ抽出のメインプロセス"""
        print("\n=== Starting data extraction ===")
        
        # 1. データ読み込み
        print("\n1. Loading PandaSet data...")
        dataparser_outputs = self._load_pandaset_data()
        
        # 2. カメラデータ抽出
        print("\n2. Extracting camera data...")
        self._extract_camera_data(dataparser_outputs)
        
        # 3. 点群データの準備
        print("\n3. Building seed points...")
        seed_points = self._build_seed_points(dataparser_outputs)
        
        # 4. 動的物体の設定
        print("\n4. Setting up dynamic actors...")
        dynamic_actors = self._setup_dynamic_actors(dataparser_outputs)
        
        # 5. 点群分離
        print("\n5. Splitting point clouds...")
        self._split_point_clouds(seed_points, dynamic_actors)
        
        # 6. 動的物体の位置姿勢履歴
        print("\n6. Extracting actor pose history...")
        self._extract_actor_poses(dynamic_actors, dataparser_outputs)
        
        # 7. メタデータ生成
        print("\n7. Generating metadata...")
        self._generate_metadata(dataparser_outputs, dynamic_actors)
        
        print("\n=== Data extraction completed ===")
    
    def _load_pandaset_data(self):
        """PandaSetデータを読み込む"""
        # データパーサー設定
        parser_config = PandaSetDataParserConfig(
            data=self.config.pandaset_path,
            sequence=self.config.sequence,
            cameras=("front", "front_left", "front_right", "back", "left", "right"),
            lidars=("Pandar64",),
        )
        
        # データパーサー作成
        dataparser = parser_config.setup()
        
        # データ生成（trainモード）
        dataparser_outputs = dataparser._generate_dataparser_outputs(split="train")
        
        print(f"  Loaded {len(dataparser_outputs.cameras)} camera frames")
        print(f"  Loaded {len(dataparser_outputs.metadata['lidars'])} lidar frames")
        
        return dataparser_outputs
    
    def _extract_camera_data(self, dataparser_outputs):
        """カメラの時間・内外パラメータを抽出"""
        cameras = dataparser_outputs.cameras
        
        self.camera_data = {
            "num_cameras": len(cameras),
            "frames": []
        }
        
        # カメラ画像保存用ディレクトリ
        camera_images_dir = self.config.output_dir / "camera_images"
        camera_images_dir.mkdir(exist_ok=True)
        
        for i in range(len(cameras)):
            frame_data = {
                "frame_id": i,
                "time": float(cameras.times[i].item()),
                "camera_to_world": cameras.camera_to_worlds[i].cpu().numpy().tolist(),
                "fx": float(cameras.fx[i].item()),
                "fy": float(cameras.fy[i].item()),
                "cx": float(cameras.cx[i].item()),
                "cy": float(cameras.cy[i].item()),
                "width": int(cameras.width[i].item()),
                "height": int(cameras.height[i].item()),
                "distortion_params": cameras.distortion_params[i].cpu().numpy().tolist() if cameras.distortion_params is not None else None,
            }
            
            # カメラ画像パスがある場合は保存
            if hasattr(dataparser_outputs, 'image_filenames') and dataparser_outputs.image_filenames is not None:
                if i < len(dataparser_outputs.image_filenames):
                    original_image_path = Path(dataparser_outputs.image_filenames[i])
                    if original_image_path.exists():
                        # 画像をコピー
                        import shutil
                        target_path = camera_images_dir / f"frame_{i:03d}.jpg"
                        shutil.copy2(original_image_path, target_path)
                        frame_data["image_path"] = str(target_path.relative_to(self.config.output_dir))
            
            self.camera_data["frames"].append(frame_data)
        
        print(f"  Extracted data for {len(self.camera_data['frames'])} camera frames")
        if camera_images_dir.exists() and any(camera_images_dir.iterdir()):
            print(f"  Saved camera images to {camera_images_dir}")
    
    def _build_seed_points(self, dataparser_outputs) -> Tuple[Tensor, Tensor, Tensor]:
        """NeuRAD-studio形式でワールド座標点群を構築（動的点群と座標系統一）"""
        # LiDAR点群をワールド座標に変換（SplatADPipelineの処理を参考）
        points_in_world = []
        points_times = []
        
        lidars = dataparser_outputs.metadata["lidars"]
        point_clouds = dataparser_outputs.metadata["point_clouds"]
        
        for l2w, pc in zip(lidars.lidar_to_worlds, point_clouds):
            # 有効距離内の点のみ使用
            returning = (
                pc[:, :3].norm(dim=-1) < lidars.valid_lidar_distance_threshold
            )
            # 【座標変換1】LiDAR座標 → World座標（NeuRAD-studio方式）
            # pc[returning, :3]: LiDAR local coordinates [N, 3]
            # l2w: LiDAR-to-World transformation matrix [4, 4] 
            # transform_points(): NeuRAD-studio's transformation function

            world_points = transform_points(pc[returning, :3], l2w)
            points_in_world.append(world_points)
            
            # 時間情報がある場合
            if "point_clouds_times" in dataparser_outputs.metadata:
                times = dataparser_outputs.metadata["point_clouds_times"]
                # 各リストの該当インデックスから時間を取得
                idx = len(points_in_world) - 1
                if idx < len(times) and times[idx] is not None:
                    points_times.append(times[idx][returning])
        
        # 全点群を結合
        points_in_world = torch.cat(points_in_world, dim=0).to(self.device)
        
        # 時間情報の処理
        if points_times:
            points_in_world_times = torch.cat(points_times, dim=0).to(self.device)
        else:
            # 時間情報がない場合は全て0
            points_in_world_times = torch.zeros(points_in_world.shape[0], device=self.device)
        
        # 色情報（LiDARなのでランダム生成）
        points_in_world_rgb = torch.rand_like(points_in_world) * 255
        
        print(f"  Built seed points: {points_in_world.shape[0]} points using NeuRAD-studio coordinate system")
        
        return (points_in_world, points_in_world_rgb, points_in_world_times)
    
    def _setup_dynamic_actors(self, dataparser_outputs) -> DynamicActors:
        """動的物体の軌跡情報を設定"""
        trajectories = dataparser_outputs.metadata.get("trajectories", [])
        
        if not trajectories:
            print("  Warning: No dynamic actors found in the dataset")
            return None
        
        # DynamicActorsの設定と作成
        config = DynamicActorsConfig(
            optimize_trajectories=False,  # 最適化は不要
        )
        
        dynamic_actors = DynamicActors(config, trajectories)
        dynamic_actors.to(self.device)
        
        print(f"  Set up {dynamic_actors.n_actors} dynamic actors")
        
        return dynamic_actors
    
    def _split_point_clouds(self, seed_points: Tuple[Tensor, Tensor, Tensor], 
                           dynamic_actors: Optional[DynamicActors]):
        """静的・動的点群に分離（SplatAD.split_seed_pointsを参考）"""
        points, colors, times = seed_points
        
        if dynamic_actors is None:
            # 動的物体がない場合は全て静的点群
            self.static_points = torch.cat([points, colors], dim=-1)
            self._sample_static_points()
            return
        
        num_actors = dynamic_actors.n_actors
        static_points = []
        dynamic_points = [[] for _ in range(num_actors)]
        unique_times = times.unique()
        
        # 各時刻ごとに処理
        for current_time in unique_times:
            time_mask = times == current_time
            current_points = points[time_mask]
            current_colors = colors[time_mask]
            static_mask = torch.ones(current_points.shape[0], dtype=torch.bool, device=self.device)
            
            # 動的アクターのワールド座標変換行列を取得
            # デバイス整合性のため、current_timeをdynamic_actorsのパラメータと同じデバイスに移動
            actor_device = next(dynamic_actors.parameters()).device
            current_time_on_device = current_time.to(actor_device)
            boxes2world, exists_at_time = dynamic_actors.get_boxes2world(
                current_time_on_device.unsqueeze(-1), flatten=False
            )
            boxes2world = boxes2world.squeeze(0)
            exists_at_time = exists_at_time.squeeze(0)
            
            # 各アクターごとに点群を分離
            for actor_idx in range(num_actors):
                if exists_at_time[actor_idx]:
                    # アクターのbounding box内の点を検索
                    actor_mask = points_in_box(
                        current_points,
                        boxes2world[actor_idx],
                        dynamic_actors.actor_sizes[actor_idx] + dynamic_actors.actor_padding,
                    )
                    
                    if actor_mask.any():
                        # 【座標変換2】World座標 → Actor Local座標（NeuRAD-studio方式）
                        # current_points[actor_mask]: World coordinates [N, 3] (from 座標変換1)
                        # boxes2world[actor_idx]: Actor-to-World transformation matrix [3, 4] or [4, 4]
                        # world2box: World-to-Actor transformation matrix (inverted)
                        # transform_points(): NeuRAD-studio's transformation function
                        world2box = pose_inverse(boxes2world[actor_idx]).reshape(-1, 3, 4)
                        points_in_local_box = transform_points(
                            current_points[actor_mask].reshape(-1, 3), world2box
                        )
                        
                        # 色情報と結合
                        actor_colors = current_colors[actor_mask]
                        actor_point_cloud = torch.cat([points_in_local_box, actor_colors], dim=-1)
                        dynamic_points[actor_idx].append(actor_point_cloud)
                        
                        # 静的マスクから除外
                        static_mask = static_mask & ~actor_mask
            
            # 静的点群に追加
            static_point_cloud = torch.cat([
                current_points[static_mask], 
                current_colors[static_mask]
            ], dim=-1)
            static_points.append(static_point_cloud)
        
        # 結合とサンプリング
        self.static_points = torch.cat(static_points, dim=0)
        self._sample_static_points()
        
        for actor_idx in range(num_actors):
            if dynamic_points[actor_idx]:
                actor_points = torch.cat(dynamic_points[actor_idx], dim=0)
                self.dynamic_points[actor_idx] = self._sample_points(
                    actor_points, self.config.dynamic_points_per_actor
                )
        
        print(f"  Split into {self.static_points.shape[0]} static points")
        print(f"  and {len(self.dynamic_points)} dynamic actors")
    
    def _sample_static_points(self):
        """静的点群を指定数にサンプリング"""
        self.static_points = self._sample_points(
            self.static_points, self.config.static_points_count
        )
    
    def _sample_points(self, points: Tensor, target_count: int) -> Tensor:
        """点群を指定数にサンプリング（SplatAD.prune_seed_pointsを参考）"""
        if points.shape[0] <= target_count:
            return points
        
        # ランダムサンプリング
        perm_idx = torch.randperm(points.shape[0])
        sampled_points = points[perm_idx[:target_count]]
        
        return sampled_points
    
    def _extract_actor_poses(self, dynamic_actors: Optional[DynamicActors], 
                            dataparser_outputs):
        """各動的物体の時系列位置姿勢を抽出"""
        if dynamic_actors is None:
            return
        
        # 全時刻を取得（LiDARメタデータから）
        lidars_metadata = dataparser_outputs.metadata["lidars"]
        all_times = lidars_metadata.times.unique()
        
        for actor_idx in range(dynamic_actors.n_actors):
            poses_list = []
            
            for time in all_times:
                # 各時刻での位置姿勢を取得
                # デバイス整合性のため、timeをdynamic_actorsのパラメータと同じデバイスに移動
                actor_device = next(dynamic_actors.parameters()).device
                time_on_device = time.to(actor_device)
                boxes2world, exists = dynamic_actors.get_boxes2world(
                    time_on_device.unsqueeze(-1), flatten=False
                )
                boxes2world = boxes2world.squeeze(0)
                exists = exists.squeeze(0)
                
                if exists[actor_idx]:
                    pose_matrix = boxes2world[actor_idx].cpu().numpy()
                    poses_list.append({
                        "time": float(time.item()),
                        "pose": pose_matrix.tolist(),
                        "position": pose_matrix[:3, 3].tolist(),
                        "rotation": pose_matrix[:3, :3].tolist(),
                    })
            
            self.actor_poses[actor_idx] = {
                "actor_id": actor_idx,
                "num_poses": len(poses_list),
                "poses": poses_list,
                "size": dynamic_actors.actor_sizes[actor_idx].cpu().numpy().tolist(),
            }
        
        print(f"  Extracted poses for {len(self.actor_poses)} actors")
    
    def _generate_metadata(self, dataparser_outputs, dynamic_actors: Optional[DynamicActors]):
        """メタデータを生成"""
        self.metadata = {
            "sequence": self.config.sequence,
            "num_lidar_frames": len(dataparser_outputs.metadata["lidars"].times),
            "num_camera_frames": len(dataparser_outputs.cameras),
            "static_points_count": self.static_points.shape[0],
            "target_static_points": self.config.static_points_count,
            "num_dynamic_actors": len(self.dynamic_points),
            "target_points_per_actor": self.config.dynamic_points_per_actor,
            "dynamic_actors_info": {},
        }
        
        # 各動的物体の情報
        for actor_idx, points in self.dynamic_points.items():
            self.metadata["dynamic_actors_info"][str(actor_idx)] = {
                "points_count": points.shape[0],
                "num_poses": len(self.actor_poses.get(actor_idx, {}).get("poses", [])),
            }
    
    def _save_points_as_ply(self, points: torch.Tensor, colors: torch.Tensor, filepath: Path):
        """点群をPLY形式で保存"""
        points_np = points.cpu().numpy().astype(np.float32)
        colors_np = colors.cpu().numpy().astype(np.uint8)
        
        # PLY用の構造化配列を作成
        vertices = np.empty(len(points_np), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        vertices['x'] = points_np[:, 0]
        vertices['y'] = points_np[:, 1]
        vertices['z'] = points_np[:, 2]
        vertices['red'] = colors_np[:, 0]
        vertices['green'] = colors_np[:, 1]
        vertices['blue'] = colors_np[:, 2]
        
        # PLYエレメントを作成
        ply_element = PlyElement.describe(vertices, 'vertex')
        
        # PLYファイルとして保存
        PlyData([ply_element]).write(str(filepath))

    def save_extracted_data(self):
        """抽出したデータを保存"""
        print("\n=== Saving extracted data ===")
        
        # 1. 静的点群をPLY形式で保存
        static_path = self.config.output_dir / "static_points.ply"
        self._save_points_as_ply(
            self.static_points[:, :3], 
            self.static_points[:, 3:], 
            static_path
        )
        print(f"  Saved static points to {static_path}")
        
        # 2. 動的点群をPLY形式で保存
        for actor_idx, points in self.dynamic_points.items():
            dynamic_path = self.config.output_dir / "dynamic_points" / f"actor_{actor_idx}.ply"
            self._save_points_as_ply(
                points[:, :3], 
                points[:, 3:], 
                dynamic_path
            )
            print(f"  Saved dynamic points for actor {actor_idx} to {dynamic_path}")
        
        # 3. アクターの位置姿勢を保存
        poses_path = self.config.output_dir / "actor_poses.json"
        with open(poses_path, "w") as f:
            json.dump(self.actor_poses, f, indent=2)
        print(f"  Saved actor poses to {poses_path}")
        
        # 4. カメラデータを保存
        camera_path = self.config.output_dir / "camera_data.json"
        with open(camera_path, "w") as f:
            json.dump(self.camera_data, f, indent=2)
        print(f"  Saved camera data to {camera_path}")
        
        # 5. メタデータを保存
        metadata_path = self.config.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  Saved metadata to {metadata_path}")
        
        print("\n=== All data saved successfully ===")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract static/dynamic point clouds from PandaSet")
    parser.add_argument("--pandaset-path", type=str, required=True, help="Path to PandaSet dataset")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name to process")
    parser.add_argument("--output-dir", type=str, default="./extracted_data", help="Output directory")
    parser.add_argument("--static-points", type=int, default=80000, help="Number of static points")
    parser.add_argument("--dynamic-points", type=int, default=5000, help="Points per dynamic actor")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 設定作成
    config = ExtractionConfig(
        pandaset_path=Path(args.pandaset_path),
        sequence=args.sequence,
        output_dir=Path(args.output_dir),
        static_points_count=args.static_points,
        dynamic_points_per_actor=args.dynamic_points,
        device=args.device,
    )
    
    # データ抽出実行
    extractor = PandaSetDataExtractor(config)
    extractor.extract_data()
    extractor.save_extracted_data()


if __name__ == "__main__":
    main()