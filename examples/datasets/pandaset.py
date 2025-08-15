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
        visibility_margin: float = 20.0,  # 視野外マージン（メートル）
    ):
        """
        Args:
            data_dir: 抽出データディレクトリパス
            split: "train", "val", "test"
            test_every: N個に1個のフレームをテスト用に使用
            patch_size: パッチ学習用のサイズ（実験的）
            load_depths: 深度情報を読み込むか（現在未対応）
            device: Tensorデバイス
            visibility_margin: 視野判定のマージン（学習中の動きを考慮）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.test_every = test_every
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.device = device
        self.visibility_margin = visibility_margin
        
        # データ読み込み
        self._load_metadata()
        self._load_static_points()
        self._load_dynamic_points()
        self._load_actor_poses()
        self._load_camera_data()
        self._split_frames()
        
        # 各フレームでの可視アクターと補間情報を事前計算
        self._precompute_frame_actors()
        
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
    
    def _check_actor_visibility(self, actor_id: int, time: float, 
                               camtoworld: np.ndarray, K: np.ndarray,
                               width: int, height: int) -> bool:
        """
        アクターがカメラ視野内にあるかチェック（マージン付き）
        
        Args:
            actor_id: アクターID
            time: 時刻
            camtoworld: カメラ→ワールド変換行列 [4, 4]
            K: カメラ内部パラメータ [3, 3]
            width, height: 画像サイズ
            
        Returns:
            bool: 視野内（またはマージン内）にある場合True
        """
        if actor_id not in self.dynamic_points:
            return False
        
        # アクター点群をワールド座標に変換
        pose_matrix = self.interpolate_actor_pose(actor_id, time)
        local_points = self.dynamic_points[actor_id][:, :3]  # [N, 3]
        
        # 代表点を使用（重心のみ）
        center = local_points.mean(axis=0)
        
        # 代表点をワールド座標に変換（centerのみ使用）
        test_points = center.reshape(1, 3)  # [1, 3]
        test_points_h = np.hstack([test_points, np.ones((len(test_points), 1))])  # [1, 4]
        world_points = (pose_matrix @ test_points_h.T).T[:, :3]  # [1, 3]
        
        # カメラ座標系に変換
        worldtocam = np.linalg.inv(camtoworld)
        cam_points = (worldtocam[:3, :3] @ world_points.T + worldtocam[:3, 3:4]).T  # [1, 3]
        
        
        # Z > 0（カメラ前方）のチェック
        if np.all(cam_points[:, 2] <= 0):
            return False
        
        # 有効な点のみを投影
        valid_mask = cam_points[:, 2] > 0
        valid_cam_points = cam_points[valid_mask]
        
        if len(valid_cam_points) == 0:
            return False
        
        # 画像平面に投影
        proj_points = valid_cam_points @ K.T  # [N_valid, 3]
        proj_points = proj_points[:, :2] / proj_points[:, 2:3]  # [N_valid, 2]
        
        # 画像範囲＋マージンのチェック
        margin_pixels = self.visibility_margin * K[0, 0] / valid_cam_points[:, 2].mean()  # 深度に応じたピクセルマージン
        
        in_bounds = (
            (proj_points[:, 0] >= -margin_pixels) & 
            (proj_points[:, 0] < width + margin_pixels) &
            (proj_points[:, 1] >= -margin_pixels) & 
            (proj_points[:, 1] < height + margin_pixels)
        )
        
        return np.any(in_bounds)
    
    def _get_interpolation_info(self, actor_id: int, target_time: float) -> Dict:
        """
        アクターの補間情報を取得
        
        Args:
            actor_id: アクターID
            target_time: 対象時刻
            
        Returns:
            補間情報の辞書:
            - indices: 補間に使用する姿勢のインデックス [2]
            - times: 補間に使用する時刻 [2]
            - alpha: 補間係数（0: 最初の姿勢, 1: 次の姿勢）
        """
        if actor_id not in self.actor_poses:
            return {
                "indices": [0, 0],
                "times": [target_time, target_time],
                "alpha": 0.0,
                "valid": False
            }
        
        actor_data = self.actor_poses[actor_id]
        times = actor_data["times"]
        
        if len(times) == 0:
            return {
                "indices": [0, 0],
                "times": [target_time, target_time],
                "alpha": 0.0,
                "valid": False
            }
        
        # 時刻範囲外の場合
        if target_time <= times[0]:
            return {
                "indices": [0, 0],
                "times": [times[0], times[0]],
                "alpha": 0.0,
                "valid": True
            }
        if target_time >= times[-1]:
            last_idx = len(times) - 1
            return {
                "indices": [last_idx, last_idx],
                "times": [times[-1], times[-1]],
                "alpha": 0.0,
                "valid": True
            }
        
        # 線形補間用のインデックスを見つける
        idx = np.searchsorted(times, target_time)
        
        # 補間係数を計算
        t0, t1 = times[idx-1], times[idx]
        alpha = (target_time - t0) / (t1 - t0)
        
        return {
            "indices": [idx-1, idx],
            "times": [t0, t1],
            "alpha": float(alpha),
            "valid": True
        }
    
    def _precompute_frame_actors(self):
        """
        各フレームでの可視アクターと補間情報を事前計算
        """
        print("Precomputing visible actors for each frame...")
        self.frame_actor_info = {}
        
        for frame_idx in range(len(self.all_frames)):
            frame_data = self.all_frames[frame_idx]
            time = frame_data["time"]
            width, height = int(frame_data["width"]), int(frame_data["height"])
            
            # カメラパラメータ
            K = np.array([
                [frame_data["fx"], 0, frame_data["cx"]],
                [0, frame_data["fy"], frame_data["cy"]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # カメラ外部パラメータ
            camera_to_world_gl = np.array(frame_data["camera_to_world"], dtype=np.float32)
            if camera_to_world_gl.shape == (3, 4):
                c2w_gl = np.eye(4, dtype=np.float32)
                c2w_gl[:3, :] = camera_to_world_gl
            else:
                c2w_gl = camera_to_world_gl
            
            # OpenGL→OpenCV座標変換
            gl_to_cv = np.eye(4, dtype=np.float32)
            gl_to_cv[1, 1] = -1.0
            gl_to_cv[2, 2] = -1.0
            camtoworld = c2w_gl @ gl_to_cv
            
            # 可視アクターをチェック
            visible_actors = {}
            for actor_id in self.dynamic_points.keys():
                if self._check_actor_visibility(actor_id, time, camtoworld, K, width, height):
                    # 補間情報を取得
                    interp_info = self._get_interpolation_info(actor_id, time)
                    visible_actors[actor_id] = interp_info
            
            self.frame_actor_info[frame_idx] = {
                "time": time,
                "visible_actors": visible_actors,
                "camtoworld": camtoworld,  # 保存して再利用
                "K": K,
                "width": width,
                "height": height
            }
            
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{len(self.all_frames)}: {len(visible_actors)} visible actors")
        
        print(f"Precomputation complete. Average visible actors: "
              f"{np.mean([len(info['visible_actors']) for info in self.frame_actor_info.values()]):.1f}")
    
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
                "actor_interpolation": Dict[int, Dict],  # actor_id -> interpolation info
            }
        """
        frame_idx = self.frame_indices[idx]
        
        # 事前計算された情報を取得
        frame_info = self.frame_actor_info[frame_idx]
        
        # 保存されたカメラパラメータを使用
        camtoworld = torch.from_numpy(frame_info["camtoworld"])
        K = torch.from_numpy(frame_info["K"])
        width = frame_info["width"]
        height = frame_info["height"]
        time = frame_info["time"]
        
        # カメラ画像の読み込み
        image = self._load_camera_image(frame_idx, width, height)
        
        # 可視アクターと補間情報
        actor_interpolation = frame_info["visible_actors"]
        
        return {
            "frame_id": frame_idx,
            "time": time,
            "image": image,
            "camtoworld": camtoworld,
            "K": K,
            "width": width,
            "height": height,
            "actor_interpolation": actor_interpolation,
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