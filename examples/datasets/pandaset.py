"""
PandaSet dataset loader for dynamic Gaussian Splatting

このモジュールは動的物体を含むPandaSetデータセットを読み込み、
静的環境と動的物体を統合したGaussian Splatting用のデータを提供します。

主要機能：
1. 静的点群の読み込み（ワールド座標）
2. 動的物体点群の読み込み（アクターローカル座標）
3. アクター姿勢の時間補間
4. カメラパラメータの座標変換（OpenGL→OpenCV）
5. フレーム単位でのデータ提供
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# PLY読み込み用
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Warning: plyfile not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement


class PandaSetDataset(Dataset):
    """
    PandaSet抽出データ用のデータセットクラス
    
    データ構造：
    - static_points.ply: 静的環境点群（ワールド座標、RGB付き）
    - dynamic_points/actor_*.ply: 各動的物体の点群（ローカル座標、RGB付き）
    - actor_poses.json: 各フレーム・アクターの姿勢（4x4変換行列）
    - camera_data.json: カメラパラメータとフレーム情報
    - metadata.json: データセットメタ情報
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        test_every: int = 8,
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            data_dir: 抽出データディレクトリパス
            split: "train", "val", "test"
            test_every: N個に1個のフレームをテスト用に使用
            patch_size: パッチ学習用のサイズ（実験的）
            load_depths: 深度情報を読み込むか（現在未対応）
            device: Tensorデバイス
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.test_every = test_every
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.device = device
        
        # データ読み込み
        self._load_metadata()
        self._load_static_points()
        self._load_dynamic_points()
        self._load_actor_poses()
        self._load_camera_data()
        self._split_frames()
        
        print(f"PandaSetDataset loaded:")
        print(f"  Static points: {len(self.static_points)}")
        print(f"  Dynamic actors: {len(self.dynamic_points)}")
        print(f"  Total frames: {len(self.all_frames)}")
        print(f"  {split} frames: {len(self.frame_indices)}")
    
    def _load_metadata(self):
        """メタデータの読み込み"""
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        print(f"Dataset: {self.metadata.get('sequence', 'unknown')}")
        print(f"Frames: {self.metadata.get('num_frames', 'unknown')}")
    
    def _load_static_points(self):
        """静的点群の読み込み（ワールド座標系）"""
        static_ply_path = self.data_dir / "static_points.ply"
        
        if not static_ply_path.exists():
            raise FileNotFoundError(f"Static points file not found: {static_ply_path}")
        
        ply_data = PlyData.read(str(static_ply_path))
        vertices = ply_data['vertex']
        
        # [N, 6] 形式: [x, y, z, r, g, b]
        self.static_points = np.column_stack([
            vertices['x'], vertices['y'], vertices['z'],
            vertices['red'] / 255.0, vertices['green'] / 255.0, vertices['blue'] / 255.0
        ]).astype(np.float32)
        
        print(f"Loaded {len(self.static_points)} static points")
    
    def _load_dynamic_points(self):
        """動的物体点群の読み込み（各アクターのローカル座標系）"""
        dynamic_dir = self.data_dir / "dynamic_points"
        self.dynamic_points = {}
        
        if not dynamic_dir.exists():
            print("Warning: No dynamic points directory found")
            return
        
        for ply_file in dynamic_dir.glob("actor_*.ply"):
            # ファイル名からアクターIDを抽出
            actor_id = int(ply_file.stem.split('_')[1])
            
            ply_data = PlyData.read(str(ply_file))
            vertices = ply_data['vertex']
            
            # [N, 6] 形式: [x, y, z, r, g, b] （アクターローカル座標）
            actor_points = np.column_stack([
                vertices['x'], vertices['y'], vertices['z'],
                vertices['red'] / 255.0, vertices['green'] / 255.0, vertices['blue'] / 255.0
            ]).astype(np.float32)
            
            self.dynamic_points[actor_id] = actor_points
            print(f"Loaded actor {actor_id}: {len(actor_points)} points")
    
    def _load_actor_poses(self):
        """アクター姿勢データの読み込み"""
        poses_path = self.data_dir / "actor_poses.json"
        
        if not poses_path.exists():
            print("Warning: No actor poses file found")
            self.actor_poses = {}
            return
        
        with open(poses_path) as f:
            raw_poses = json.load(f)
        
        # アクターID毎に時刻順にソートされた姿勢データを保持
        self.actor_poses = {}
        for actor_id_str, actor_data in raw_poses.items():
            actor_id = int(actor_id_str)
            poses = actor_data["poses"]
            
            # 時刻順にソート
            poses.sort(key=lambda p: p["time"])
            
            self.actor_poses[actor_id] = {
                "poses": poses,
                "times": np.array([p["time"] for p in poses]),
                "matrices": np.array([p["pose"] for p in poses])  # [T, 4, 4]
            }
            
            print(f"Loaded poses for actor {actor_id}: {len(poses)} poses")
    
    def _load_camera_data(self):
        """カメラデータの読み込み"""
        camera_path = self.data_dir / "camera_data.json"
        
        with open(camera_path) as f:
            camera_data = json.load(f)
        
        self.all_frames = camera_data["frames"]
        print(f"Loaded {len(self.all_frames)} camera frames")
    
    def _split_frames(self):
        """フレームを訓練/検証/テスト用に分割"""
        num_frames = len(self.all_frames)
        
        if self.split == "train":
            # test_every毎にスキップしてフレームを選択
            self.frame_indices = [i for i in range(num_frames) if i % self.test_every != 0]
        elif self.split == "val" or self.split == "test":
            # test_every毎のフレームを使用
            self.frame_indices = [i for i in range(num_frames) if i % self.test_every == 0]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def interpolate_actor_pose(self, actor_id: int, target_time: float) -> np.ndarray:
        """
        指定時刻でのアクター姿勢を補間
        
        Args:
            actor_id: アクターID
            target_time: 対象時刻
            
        Returns:
            pose_matrix: [4, 4] 変換行列（アクター→ワールド座標）
        """
        if actor_id not in self.actor_poses:
            # 姿勢情報がない場合は単位行列を返す
            return np.eye(4, dtype=np.float32)
        
        actor_data = self.actor_poses[actor_id]
        times = actor_data["times"]
        matrices = actor_data["matrices"]
        
        if len(times) == 0:
            return np.eye(4, dtype=np.float32)
        
        # 時刻範囲外の場合は最も近い姿勢を使用
        if target_time <= times[0]:
            return matrices[0].astype(np.float32)
        if target_time >= times[-1]:
            return matrices[-1].astype(np.float32)
        
        # 線形補間用のインデックスを見つける
        idx = np.searchsorted(times, target_time)
        if idx == 0:
            return matrices[0].astype(np.float32)
        
        # 時刻での線形補間
        t0, t1 = times[idx-1], times[idx]
        alpha = (target_time - t0) / (t1 - t0)
        
        # 変換行列の補間（簡単な線形補間）
        # 注意: 正確には回転はSLERP、平行移動は線形補間すべき
        pose_interp = (1 - alpha) * matrices[idx-1] + alpha * matrices[idx]
        
        return pose_interp.astype(np.float32)
    
    def transform_dynamic_points_to_world(self, actor_id: int, time: float) -> np.ndarray:
        """
        動的物体の点群を指定時刻のワールド座標に変換
        
        Args:
            actor_id: アクターID
            time: 時刻
            
        Returns:
            world_points: [N, 6] ワールド座標の点群 [x, y, z, r, g, b]
        """
        if actor_id not in self.dynamic_points:
            return np.empty((0, 6), dtype=np.float32)
        
        # アクターローカル座標の点群
        local_points = self.dynamic_points[actor_id]
        points_xyz = local_points[:, :3]  # [N, 3]
        colors = local_points[:, 3:]      # [N, 3]
        
        # アクター姿勢を取得
        pose_matrix = self.interpolate_actor_pose(actor_id, time)
        
        # ローカル座標→ワールド座標変換
        points_homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])  # [N, 4]
        world_points_homogeneous = points_homogeneous @ pose_matrix.T  # [N, 4]
        world_points = world_points_homogeneous[:, :3]  # [N, 3]
        
        return np.hstack([world_points, colors])  # [N, 6]
    
    def get_combined_points_for_time(self, time: float) -> np.ndarray:
        """
        指定時刻での静的+動的統合点群を取得
        
        Args:
            time: 時刻
            
        Returns:
            combined_points: [N, 6] 統合点群 [x, y, z, r, g, b]
        """
        points_list = [self.static_points]  # 静的点群
        
        # 各動的アクターの点群を変換して追加
        for actor_id in self.dynamic_points.keys():
            world_points = self.transform_dynamic_points_to_world(actor_id, time)
            if len(world_points) > 0:
                points_list.append(world_points)
        
        if len(points_list) == 1:
            return points_list[0]
        else:
            return np.vstack(points_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        フレームデータを取得
        
        Returns:
            data: {
                "frame_id": int,
                "time": float,
                "image": torch.Tensor [H, W, 3],
                "camtoworld": torch.Tensor [4, 4],
                "K": torch.Tensor [3, 3],
                "width": int,
                "height": int,
                "static_points": torch.Tensor [N_static, 6],
                "dynamic_points": Dict[int, torch.Tensor],  # actor_id -> [N_actor, 6] (local coords)
                "actor_poses": Dict[int, torch.Tensor],     # actor_id -> [4, 4] world pose
            }
        """
        frame_idx = self.frame_indices[idx]
        frame_data = self.all_frames[frame_idx]
        
        # 基本情報
        time = frame_data["time"]
        width, height = int(frame_data["width"]), int(frame_data["height"])
        
        # カメラ内部パラメータ
        K = torch.tensor([
            [frame_data["fx"], 0, frame_data["cx"]],
            [0, frame_data["fy"], frame_data["cy"]],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # カメラ外部パラメータ（OpenGL→OpenCV座標系変換）
        camera_to_world_gl = np.array(frame_data["camera_to_world"], dtype=np.float32)
        if camera_to_world_gl.shape == (3, 4):
            # 3x4 -> 4x4
            c2w_gl = np.eye(4, dtype=np.float32)
            c2w_gl[:3, :] = camera_to_world_gl
        else:
            c2w_gl = camera_to_world_gl
        
        # OpenGL→OpenCV座標変換（Y軸、Z軸反転）
        gl_to_cv = np.eye(4, dtype=np.float32)
        gl_to_cv[1, 1] = -1.0  # Y軸反転
        gl_to_cv[2, 2] = -1.0  # Z軸反転
        
        c2w_cv = c2w_gl @ gl_to_cv
        camtoworld = torch.from_numpy(c2w_cv)
        
        # カメラ画像の読み込み（デモ用：グレー画像生成）
        # 実際のプロジェクトでは extracted_data/camera_images から読み込み
        image = self._load_camera_image(frame_idx, width, height)
        
        # 静的点群
        static_points = torch.from_numpy(self.static_points)
        
        # 動的点群（ローカル座標）
        dynamic_points = {}
        for actor_id, points in self.dynamic_points.items():
            dynamic_points[actor_id] = torch.from_numpy(points)
        
        # アクター姿勢
        actor_poses = {}
        for actor_id in self.dynamic_points.keys():
            pose = self.interpolate_actor_pose(actor_id, time)
            actor_poses[actor_id] = torch.from_numpy(pose)
        
        return {
            "frame_id": frame_idx,
            "time": time,
            "image": image,
            "camtoworld": camtoworld,
            "K": K,
            "width": width,
            "height": height,
            "static_points": static_points,
            "dynamic_points": dynamic_points,
            "actor_poses": actor_poses,
        }
    
    def _load_camera_image(self, frame_id: int, width: int, height: int) -> torch.Tensor:
        """
        カメラ画像を読み込み（現在はデモ用のダミー画像）
        
        実際の実装では extracted_data/camera_images/frame_{frame_id:03d}.jpg を読み込み
        """
        # カメラ画像ディレクトリの確認
        camera_images_dir = self.data_dir / "camera_images"
        image_path = camera_images_dir / f"frame_{frame_id:03d}.jpg"
        
        if image_path.exists():
            # 実際の画像を読み込み
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(image.astype(np.float32))
        else:
            # デモ用：グレーのグラデーション画像を生成
            image = np.full((height, width, 3), 128, dtype=np.uint8)
            
            # フレーム情報を描画
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Frame {frame_id}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            
            return torch.from_numpy(image.astype(np.float32))


# テスト用のデータセット作成関数
def create_demo_dataset(data_dir: Path, num_frames: int = 100):
    """
    デモ用のPandaSetデータセットを作成
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # デモ用の静的点群
    static_points = np.random.rand(1000, 3) * 20 - 10  # -10 to 10の範囲
    static_colors = np.random.rand(1000, 3) * 255
    
    # PLYファイルとして保存
    # ... (実装省略)
    
    print(f"Demo dataset created at {data_dir}")


if __name__ == "__main__":
    # テスト実行
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Extracted data directory")
    args = parser.parse_args()
    
    # データセットのテスト読み込み
    dataset = PandaSetDataset(args.data_dir, split="train")
    
    print(f"Dataset size: {len(dataset)}")
    
    # 最初のフレームをテスト
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Static points shape: {sample['static_points'].shape}")
    print(f"Dynamic actors: {len(sample['dynamic_points'])}")
    print(f"Image shape: {sample['image'].shape}")