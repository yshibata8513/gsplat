"""
Dynamic Gaussian Splats management system

動的物体を含むGaussian Splatsの管理システム。
静的環境と動的物体（アクター）を統合して扱い、
アクターの位置・姿勢変化に対応したレンダリングを可能にする。

主要機能：
1. 静的ガウシアンの管理
2. アクター別動的ガウシアンの管理
3. アクター姿勢パラメータの管理
4. 統合レンダリング用の座標変換
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/workspace/gsplat/examples')

from utils.quaternion_utils import (
    quaternion_multiply, 
    quaternion_multiply_batch,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)


class ActorPoseParams(nn.Module):
    """
    動的アクターの姿勢パラメータ管理
    
    各アクターについて時系列の位置・回転を学習可能パラメータとして保持し、
    任意の時刻での姿勢を補間して取得可能。
    """
    
    def __init__(
        self,
        actor_id: int,
        pose_times: torch.Tensor,  # [T,] 時刻リスト
        initial_poses: torch.Tensor,  # [T, 4, 4] 初期姿勢行列
        learn_poses: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            actor_id: アクターID
            pose_times: 姿勢が定義される時刻リスト [T,]
            initial_poses: 初期姿勢変換行列 [T, 4, 4]
            learn_poses: 姿勢を学習するかどうか
            device: デバイス
        """
        super().__init__()
        
        self.actor_id = actor_id
        self.device = device
        self.learn_poses = learn_poses
        
        # 時刻情報（固定）
        self.register_buffer("pose_times", pose_times.to(device))
        
        # 姿勢パラメータを平行移動と回転に分解
        translations = initial_poses[:, :3, 3]  # [T, 3]
        rotation_matrices = initial_poses[:, :3, :3]  # [T, 3, 3]
        
        # 回転行列をクォータニオンに変換
        quaternions = torch.stack([
            rotation_matrix_to_quaternion(R) for R in rotation_matrices
        ])  # [T, 4]
        
        if learn_poses:
            # 学習可能パラメータとして設定
            self.translations = nn.Parameter(translations.to(device))
            self.quaternions = nn.Parameter(quaternions.to(device))
        else:
            # 固定パラメータとして設定
            self.register_buffer("translations", translations.to(device))
            self.register_buffer("quaternions", quaternions.to(device))
        
        print(f"ActorPoseParams[{actor_id}]: {len(pose_times)} poses, learn={learn_poses}")
    
    def interpolate_pose(self, target_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定時刻での姿勢を補間
        
        Args:
            target_time: 対象時刻
            
        Returns:
            translation: [3,] 平行移動ベクトル
            quaternion: [4,] 回転クォータニオン（正規化済み）
        """
        times = self.pose_times
        
        if len(times) == 1:
            # 1つしか姿勢がない場合
            return self.translations[0], F.normalize(self.quaternions[0], dim=-1)
        
        # 時刻範囲のクランプ
        if target_time <= times[0]:
            return self.translations[0], F.normalize(self.quaternions[0], dim=-1)
        if target_time >= times[-1]:
            return self.translations[-1], F.normalize(self.quaternions[-1], dim=-1)
        
        # 補間用インデックスを見つける
        idx = torch.searchsorted(times, target_time).item()
        if idx == 0:
            return self.translations[0], F.normalize(self.quaternions[0], dim=-1)
        
        # 線形補間パラメータ
        t0, t1 = times[idx-1].item(), times[idx].item()
        alpha = (target_time - t0) / (t1 - t0)
        
        # 平行移動の線形補間
        trans_interp = (1 - alpha) * self.translations[idx-1] + alpha * self.translations[idx]
        
        # クォータニオンの球面線形補間（SLERP）
        quat_interp = self._slerp(self.quaternions[idx-1], self.quaternions[idx], alpha)
        
        return trans_interp, quat_interp
    
    def _slerp(self, q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """
        球面線形補間（SLERP）
        
        Args:
            q1, q2: 補間するクォータニオン [4,]
            t: 補間パラメータ [0, 1]
            
        Returns:
            interpolated_quat: 補間されたクォータニオン [4,]
        """
        # クォータニオンを正規化
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        
        # 内積を計算
        dot = torch.sum(q1 * q2)
        
        # 最短経路を選ぶため、内積が負なら q2 を反転
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # ほぼ同じ方向の場合は線形補間
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return F.normalize(result, dim=-1)
        
        # SLERP計算
        theta_0 = torch.acos(torch.clamp(torch.abs(dot), 0, 1))
        sin_theta_0 = torch.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def get_pose_matrix(self, target_time: float) -> torch.Tensor:
        """
        指定時刻での4x4姿勢変換行列を取得
        
        Args:
            target_time: 対象時刻
            
        Returns:
            pose_matrix: [4, 4] 変換行列（アクター→ワールド座標）
        """
        translation, quaternion = self.interpolate_pose(target_time)
        
        # クォータニオンから回転行列を生成
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # 4x4変換行列を構築
        pose_matrix = torch.eye(4, device=self.device)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation
        
        return pose_matrix


class DynamicGaussianSplats(nn.Module):
    """
    静的・動的ガウシアンの統合管理システム
    
    静的環境のガウシアンと、複数の動的アクターのガウシアンを管理し、
    アクターの姿勢変化に応じた座標変換を行って統合レンダリングを可能にする。
    """
    
    def __init__(
        self,
        static_splats: Dict[str, nn.Parameter],
        dynamic_splats: Dict[int, Dict[str, nn.Parameter]],
        actor_poses: Dict[int, ActorPoseParams],
        device: str = "cuda"
    ):
        """
        Args:
            static_splats: 静的ガウシアンパラメータ
            dynamic_splats: アクター別動的ガウシアンパラメータ
            actor_poses: アクター姿勢パラメータ
            device: デバイス
        """
        super().__init__()
        
        self.device = device
        
        # 静的ガウシアン（ワールド座標）
        self.static_splats = nn.ParameterDict(static_splats)
        
        # 動的ガウシアン（アクター別、ローカル座標）
        self.dynamic_splats = nn.ModuleDict()
        for actor_id, splats in dynamic_splats.items():
            self.dynamic_splats[str(actor_id)] = nn.ParameterDict(splats)
        
        # アクター姿勢パラメータ
        self.actor_poses = nn.ModuleDict()
        for actor_id, pose_params in actor_poses.items():
            self.actor_poses[str(actor_id)] = pose_params
        
        # 統計情報
        static_count = len(self.static_splats.get("means", []))
        dynamic_count = sum(
            len(splats.get("means", [])) for splats in self.dynamic_splats.values()
        )
        
        print(f"DynamicGaussianSplats initialized:")
        print(f"  Static gaussians: {static_count}")
        print(f"  Dynamic actors: {len(self.dynamic_splats)}")
        print(f"  Dynamic gaussians: {dynamic_count}")
        print(f"  Total: {static_count + dynamic_count}")
    
    def transform_dynamic_gaussians(
        self, 
        actor_id: int, 
        time: float
    ) -> Dict[str, torch.Tensor]:
        """
        動的アクターのガウシアンをローカル座標からワールド座標に変換
        
        Args:
            actor_id: アクターID
            time: 時刻
            
        Returns:
            transformed_splats: ワールド座標に変換されたガウシアンパラメータ
        """
        actor_key = str(actor_id)
        
        if actor_key not in self.dynamic_splats or actor_key not in self.actor_poses:
            return {}
        
        # アクターのローカル座標ガウシアン
        local_splats = self.dynamic_splats[actor_key]
        
        # 時刻での姿勢を取得
        pose_params = self.actor_poses[actor_key]
        translation, quaternion = pose_params.interpolate_pose(time)
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # 変換後のガウシアンパラメータ
        transformed = {}
        
        # 1. 位置変換: means_world = R @ means_local + t
        if "means" in local_splats:
            local_means = local_splats["means"]  # [N, 3]
            world_means = (rotation_matrix @ local_means.T).T + translation[None, :]  # [N, 3]
            transformed["means"] = world_means
        
        # 2. 回転変換: quats_world = quat_actor * quats_local
        if "quats" in local_splats:
            local_quats = F.normalize(local_splats["quats"], dim=-1)  # [N, 4]
            # 各ガウシアンの回転にアクター回転を適用（ベクトル化）
            N = local_quats.shape[0]
            actor_quat_batch = quaternion.unsqueeze(0).expand(N, -1)  # [N, 4]
            world_quats = quaternion_multiply_batch(actor_quat_batch, local_quats)
            
            # NaN/Inf check - これが数値不安定性の原因
            if torch.isnan(world_quats).any() or torch.isinf(world_quats).any():
                print(f"WARNING: NaN/Inf detected in quaternion transformation for actor {actor_id}")
                print(f"  actor_quat: {quaternion}")
                print(f"  local_quats range: [{local_quats.min():.6f}, {local_quats.max():.6f}]")
                print(f"  world_quats NaN count: {torch.isnan(world_quats).sum()}")
                print(f"  world_quats Inf count: {torch.isinf(world_quats).sum()}")
            
            transformed["quats"] = world_quats
        
        # 3. その他のパラメータは不変
        for key in ["scales", "opacities", "sh0", "shN", "features", "colors"]:
            if key in local_splats:
                transformed[key] = local_splats[key]
        
        return transformed
    
    def get_combined_gaussians(self, 
                             time: Optional[float] = None,
                             frame_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        指定時刻での静的+動的ガウシアンの統合
        
        Args:
            time: 時刻（後方互換性のため残す）
            frame_data: PandaSetDatasetの__getitem__出力
                - actor_interpolation: 可視アクターの補間情報
                - time: フレーム時刻
            
        Returns:
            combined_splats: 統合されたガウシアンパラメータ
        """
        # frame_dataが提供された場合はそこから情報を取得
        if frame_data is not None:
            time = frame_data["time"]
            visible_actors = frame_data.get("actor_interpolation", {})
        else:
            # 後方互換性：すべてのアクターを処理
            if time is None:
                raise ValueError("Either time or frame_data must be provided")
            visible_actors = {int(actor_id): None for actor_id in self.dynamic_splats.keys()}
        
        combined = {}
        
        # パラメータリストを準備
        params_list = {}
        
        # 静的ガウシアンを追加
        for key, param in self.static_splats.items():
            if key not in params_list:
                params_list[key] = []
            params_list[key].append(param)
        
        # 可視動的ガウシアンのみを変換して追加
        for actor_id, interp_info in visible_actors.items():
            actor_key = str(actor_id)
            if actor_key in self.dynamic_splats:
                # 補間情報が提供されている場合は使用（将来の拡張用）
                # 現在はtime情報のみ使用
                transformed = self.transform_dynamic_gaussians(int(actor_id), time)
                for key, param in transformed.items():
                    if key not in params_list:
                        params_list[key] = []
                    params_list[key].append(param)
        
        # パラメータを結合
        for key, param_list in params_list.items():
            if len(param_list) > 0:
                combined[key] = torch.cat(param_list, dim=0)
            else:
                # 空の場合は適切なshapeの空テンソルを作成
                if len(param_list) > 0:
                    combined[key] = torch.empty((0, param_list[0].shape[1]), 
                                              device=self.device, 
                                              dtype=param_list[0].dtype)
        
        return combined
    
    def get_static_count(self) -> int:
        """静的ガウシアン数を取得"""
        if "means" in self.static_splats:
            return len(self.static_splats["means"])
        return 0
    
    def get_dynamic_count(self, actor_id: Optional[int] = None) -> int:
        """動的ガウシアン数を取得"""
        if actor_id is not None:
            actor_key = str(actor_id)
            if actor_key in self.dynamic_splats and "means" in self.dynamic_splats[actor_key]:
                return len(self.dynamic_splats[actor_key]["means"])
            return 0
        else:
            # 全動的ガウシアン数
            total = 0
            for splats in self.dynamic_splats.values():
                if "means" in splats:
                    total += len(splats["means"])
            return total
    
    def get_total_count(self) -> int:
        """全ガウシアン数を取得"""
        return self.get_static_count() + self.get_dynamic_count()
    
    def get_actor_ids(self) -> List[int]:
        """アクターIDリストを取得"""
        return [int(actor_id) for actor_id in self.dynamic_splats.keys()]


def create_dynamic_splats_from_points(
    static_points: np.ndarray,  # [N_static, 6] ワールド座標の静的点群
    dynamic_points: Dict[int, np.ndarray],  # actor_id -> [N_actor, 6] ローカル座標の動的点群
    actor_poses: Dict[int, Dict],  # アクター姿勢情報
    init_params: Dict,  # 初期化パラメータ
    device: str = "cuda"
) -> DynamicGaussianSplats:
    """
    点群データからDynamicGaussianSplatsを作成
    
    Args:
        static_points: 静的点群 [N, 6] (x,y,z,r,g,b)
        dynamic_points: アクター別動的点群
        actor_poses: アクター姿勢データ
        init_params: 初期化パラメータ
        device: デバイス
        
    Returns:
        dynamic_splats: 初期化されたDynamicGaussianSplats
    """
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    
    # 静的ガウシアンの初期化
    static_splats = _create_gaussians_from_points_simple(
        static_points, init_params, device
    )
    
    # 動的ガウシアンの初期化
    dynamic_splats_dict = {}
    for actor_id, points in dynamic_points.items():
        dynamic_splats_dict[actor_id] = _create_gaussians_from_points_simple(
            points, init_params, device
        )
    
    # アクター姿勢パラメータの初期化
    actor_poses_dict = {}
    for actor_id, pose_data in actor_poses.items():
        if "times" in pose_data and "poses" in pose_data:
            times_tensor = torch.tensor(pose_data["times"], dtype=torch.float32)
            poses_tensor = torch.from_numpy(np.stack(pose_data["poses"])).float()
            
            actor_poses_dict[actor_id] = ActorPoseParams(
                actor_id=actor_id,
                pose_times=times_tensor,
                initial_poses=poses_tensor,
                learn_poses=True,
                device=device
            )
    
    return DynamicGaussianSplats(
        static_splats=static_splats,
        dynamic_splats=dynamic_splats_dict,
        actor_poses=actor_poses_dict,
        device=device
    )


def _create_gaussians_from_points_simple(
    points: np.ndarray,
    init_params: Dict,
    device: str = "cuda"
) -> Dict[str, torch.nn.Parameter]:
    """簡単な点群からガウシアンパラメータ作成"""
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    
    N = len(points)
    points_xyz = points[:, :3]
    colors_rgb = points[:, 3:6]
    
    # スケールの設定
    if init_params.get("fixed_scale", False):
        # 固定スケール使用
        fixed_scale = init_params.get("init_scale", 0.1)
        scales = torch.full((N, 3), np.log(fixed_scale))
    else:
        # 近傍距離からスケールを推定
        points_tensor = torch.from_numpy(points_xyz).float()
        if N >= 4:
            # KNNでスケール推定
            model = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(points_xyz)
            distances, _ = model.kneighbors(points_xyz)
            dist2_avg = torch.from_numpy(distances[:, 1:]).float().pow(2).mean(dim=-1)
        else:
            # 点数が少ない場合はデフォルト値
            dist2_avg = torch.full((N,), 0.1)
        
        scales = torch.log(torch.sqrt(dist2_avg) * init_params.get("init_scale", 1.0))
        scales = scales.unsqueeze(-1).repeat(1, 3)
    
    # ランダム回転
    quats = torch.rand((N, 4))
    quats = F.normalize(quats, dim=-1)
    
    # 不透明度
    opacities = torch.logit(torch.full((N,), init_params.get("init_opa", 0.1)))
    
    # 球面調和関数係数（簡易版）
    C0 = 0.28209479177387814
    sh0 = (torch.from_numpy(colors_rgb).float() - 0.5) / C0
    sh_degree = init_params.get("sh_degree", 3)
    shN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3))
    
    return {
        "means": torch.nn.Parameter(torch.from_numpy(points_xyz).float().to(device)),
        "scales": torch.nn.Parameter(scales.to(device)),
        "quats": torch.nn.Parameter(quats.to(device)),
        "opacities": torch.nn.Parameter(opacities.to(device)),
        "sh0": torch.nn.Parameter(sh0.unsqueeze(1).to(device)),
        "shN": torch.nn.Parameter(shN.to(device)),
    }