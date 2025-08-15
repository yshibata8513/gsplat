#!/usr/bin/env python3
"""
PandaSet Dynamic Gaussian Splatting Trainer

動的物体を含むPandaSetデータセットでの学習システム
可視アクター最適化と統合レンダリングパイプライン実装
"""

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

# PandaSet dynamic system imports
from datasets.pandaset import PandaSetDataset
from models.dynamic_splats import DynamicGaussianSplats, create_dynamic_splats_from_points
from utils.quaternion_utils import quaternion_to_rotation_matrix

from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


@dataclass
class DynamicConfig:
    """動的Gaussian Splatting設定"""
    # Dataset
    data_dir: str = "/workspace/gsplat/extracted_data_new"
    result_dir: str = "results/pandaset_dynamic"
    test_every: int = 10
    
    # Training
    max_steps: int = 5000
    batch_size: int = 1
    eval_steps: List[int] = field(default_factory=lambda: [1000, 3000, 5000])
    save_steps: List[int] = field(default_factory=lambda: [1000, 3000, 5000])
    
    # Model parameters
    init_scale: float = 0.3
    init_opa: float = 0.8
    sh_degree: int = 0  # 色は点群から設定するため0
    
    # Learning rates - static gaussians
    static_means_lr: float = 1.6e-4
    static_scales_lr: float = 1.6e-3
    static_quats_lr: float = 1.6e-3
    static_opacities_lr: float = 5e-2
    static_sh0_lr: float = 2.5e-3
    
    # Learning rates - dynamic gaussians  
    dynamic_means_lr: float = 8e-5
    dynamic_scales_lr: float = 8e-4
    dynamic_quats_lr: float = 8e-4
    dynamic_opacities_lr: float = 2.5e-2
    dynamic_sh0_lr: float = 1.25e-3
    
    # Learning rates - actor poses
    actor_trans_lr: float = 1e-3
    actor_rot_lr: float = 1e-3
    
    # Point cloud sampling
    static_sample_size: int = 400000  # 静的点群サンプリング数
    max_actors: int = 1000  # 最大アクター数（メモリ制限）
    max_points_per_actor: int = 2000  # アクター当たり最大点数
    
    # Rendering
    render_size: int = 512
    near_plane: float = 0.1
    far_plane: float = 200.0
    
    # Regularization
    pose_smooth_reg: float = 1e-3  # 姿勢時間的滑らかさ
    pose_velocity_reg: float = 1e-4  # 速度正則化
    
    # Output
    save_ply: bool = True
    disable_video: bool = False
    

class DynamicRunner:
    """動的Gaussian Splatting学習・評価エンジン"""
    
    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: DynamicConfig):
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        # Setup directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # Load PandaSet dataset
        self.dataset = PandaSetDataset(
            cfg.data_dir,
            split="train",
            test_every=cfg.test_every,
            visibility_margin=20.0
        )
        
        self.val_dataset = PandaSetDataset(
            cfg.data_dir,
            split="val", 
            test_every=cfg.test_every,
            visibility_margin=20.0
        )
        
        print(f"Dataset loaded: {len(self.dataset)} train, {len(self.val_dataset)} val frames")
        
        # Initialize dynamic gaussian system
        self._init_dynamic_system()
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        # Strategy
        self.strategy = DefaultStrategy(verbose=True)
        self.step = 0
        
    def _init_dynamic_system(self):
        """動的ガウシアンシステムの初期化"""
        print("Initializing dynamic gaussian system...")
        
        # Sample static points
        total_static = len(self.dataset.static_points)
        sample_rate = min(1.0, self.cfg.static_sample_size / total_static)
        static_indices = np.random.choice(
            total_static, 
            int(total_static * sample_rate), 
            replace=False
        )
        static_points = self.dataset.static_points[static_indices].copy()
        
        # Sample dynamic actors (first frame's visible actors as reference)
        sample_frame = self.dataset[0]
        visible_actors = sample_frame["actor_interpolation"]
        
        selected_actors = {}
        actor_count = 0
        
        for actor_id in visible_actors.keys():
            if actor_count >= self.cfg.max_actors:
                break
                
            if actor_id in self.dataset.dynamic_points:
                points = self.dataset.dynamic_points[actor_id]
                if len(points) > self.cfg.max_points_per_actor:
                    indices = np.random.choice(
                        len(points), 
                        self.cfg.max_points_per_actor, 
                        replace=False
                    )
                    selected_actors[actor_id] = points[indices].copy()
                else:
                    selected_actors[actor_id] = points.copy()
                actor_count += 1
        
        # Apply random colors to point clouds
        # Random colors for static points
        static_colors = np.random.rand(len(static_points), 3)
        static_points[:, 3:6] = static_colors
        
        # Random colors for dynamic objects  
        for actor_id, points in selected_actors.items():
            dynamic_colors = np.random.rand(len(points), 3)
            selected_actors[actor_id][:, 3:6] = dynamic_colors
        
        # Build actor poses
        actor_poses = {}
        for actor_id in selected_actors.keys():
            if actor_id in self.dataset.actor_poses:
                actor_data = self.dataset.actor_poses[actor_id]
                actor_poses[actor_id] = {
                    "times": actor_data["times"],
                    "poses": actor_data["matrices"]
                }
            else:
                # Fallback identity pose
                actor_poses[actor_id] = {
                    "times": [0.0],
                    "poses": [np.eye(4, dtype=np.float32)]
                }
        
        # Create dynamic system
        init_params = {
            "init_scale": self.cfg.init_scale,
            "init_opa": self.cfg.init_opa,
            "sh_degree": self.cfg.sh_degree,
        }
        
        self.dynamic_system = create_dynamic_splats_from_points(
            static_points=static_points,
            dynamic_points=selected_actors,
            actor_poses=actor_poses,
            init_params=init_params,
            device=self.device
        )
        
        print(f"✓ System initialized:")
        print(f"  Static gaussians: {self.dynamic_system.get_static_count()}")
        print(f"  Dynamic actors: {len(selected_actors)}")
        print(f"  Dynamic gaussians: {self.dynamic_system.get_dynamic_count()}")
        print(f"  Total: {self.dynamic_system.get_total_count()}")
        
        # Setup optimizers
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        """オプティマイザーの設定"""
        self.optimizers = {}
        
        # Static gaussian optimizers
        for name, param in self.dynamic_system.static_splats.items():
            if name == "means":
                lr = self.cfg.static_means_lr
            elif name == "scales":
                lr = self.cfg.static_scales_lr
            elif name == "quats":
                lr = self.cfg.static_quats_lr
            elif name == "opacities":
                lr = self.cfg.static_opacities_lr
            elif name == "sh0":
                lr = self.cfg.static_sh0_lr
            else:
                lr = 1e-3  # default
            
            self.optimizers[f"static_{name}"] = torch.optim.Adam(
                [param], lr=lr, eps=1e-15
            )
        
        # Dynamic gaussian optimizers
        for actor_id, splats in self.dynamic_system.dynamic_splats.items():
            for name, param in splats.items():
                if name == "means":
                    lr = self.cfg.dynamic_means_lr
                elif name == "scales":
                    lr = self.cfg.dynamic_scales_lr
                elif name == "quats":
                    lr = self.cfg.dynamic_quats_lr
                elif name == "opacities":
                    lr = self.cfg.dynamic_opacities_lr
                elif name == "sh0":
                    lr = self.cfg.dynamic_sh0_lr
                else:
                    lr = 5e-4  # default
                
                self.optimizers[f"dynamic_{actor_id}_{name}"] = torch.optim.Adam(
                    [param], lr=lr, eps=1e-15
                )
        
        # Actor pose optimizers
        for actor_id, pose_params in self.dynamic_system.actor_poses.items():
            self.optimizers[f"actor_{actor_id}_trans"] = torch.optim.Adam(
                [pose_params.translations], lr=self.cfg.actor_trans_lr, eps=1e-15
            )
            self.optimizers[f"actor_{actor_id}_rot"] = torch.optim.Adam(
                [pose_params.quaternions], lr=self.cfg.actor_rot_lr, eps=1e-15
            )
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """1学習ステップの実行"""
        import time
        step_start = time.time()
        
        # Get combined gaussians for this frame
        gaussian_start = time.time()
        combined_splats = self.dynamic_system.get_combined_gaussians(frame_data=batch)
        gaussian_time = time.time() - gaussian_start
        
        # Camera parameters
        prep_start = time.time()
        camtoworld = batch["camtoworld"].unsqueeze(0)
        K = batch["K"].unsqueeze(0) 
        image_gt = batch["image"].unsqueeze(0) / 255.0
        
        # Resize for training
        H, W = image_gt.shape[1:3]
        if H != self.cfg.render_size or W != self.cfg.render_size:
            image_gt = F.interpolate(
                image_gt.permute(0, 3, 1, 2),
                size=(self.cfg.render_size, self.cfg.render_size),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Adjust camera intrinsics
            K = K.clone()
            K[:, 0, 2] *= self.cfg.render_size / W  # cx
            K[:, 1, 2] *= self.cfg.render_size / H  # cy
            K[:, 0, 0] *= self.cfg.render_size / W  # fx
            K[:, 1, 1] *= self.cfg.render_size / H  # fy
        prep_time = time.time() - prep_start
        
        # Rasterization
        render_start = time.time()
        renders, alphas, info = rasterization(
            means=combined_splats["means"],
            quats=F.normalize(combined_splats["quats"], dim=-1),
            scales=torch.exp(combined_splats["scales"]),
            opacities=torch.sigmoid(combined_splats["opacities"]),
            colors=combined_splats["sh0"],
            viewmats=torch.linalg.inv(camtoworld),
            Ks=K,
            width=self.cfg.render_size,
            height=self.cfg.render_size,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        render_time = time.time() - render_start
        
        # Loss computation
        loss_start = time.time()
        l1_loss = F.l1_loss(renders, image_gt)
        ssim_loss = 1.0 - fused_ssim(renders.permute(0, 3, 1, 2), image_gt.permute(0, 3, 1, 2))
        loss = l1_loss + 0.2 * ssim_loss
        
        # Regularization losses
        reg_loss = 0.0
        
        # Pose smoothness regularization
        if self.cfg.pose_smooth_reg > 0:
            for actor_id, pose_params in self.dynamic_system.actor_poses.items():
                if len(pose_params.pose_times) > 1:
                    # Translation smoothness
                    trans_diff = pose_params.translations[1:] - pose_params.translations[:-1]
                    reg_loss += self.cfg.pose_smooth_reg * (trans_diff ** 2).mean()
                    
                    # Rotation smoothness (quaternion)
                    quat_diff = pose_params.quaternions[1:] - pose_params.quaternions[:-1]
                    reg_loss += self.cfg.pose_smooth_reg * (quat_diff ** 2).mean()
        
        total_loss = loss + reg_loss
        loss_time = time.time() - loss_start
        total_time = time.time() - step_start
        
        return {
            "loss": total_loss.item(),
            "l1_loss": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "reg_loss": reg_loss.item(),
            "psnr": -10 * math.log10(F.mse_loss(renders, image_gt).item()),
            "n_gaussians": len(combined_splats["means"]),
            # Timing info
            "time_gaussian": gaussian_time,
            "time_prep": prep_time,
            "time_render": render_time, 
            "time_loss": loss_time,
            "time_total": total_time,
        }
    
    def train(self):
        """学習メインループ"""
        print("Starting training...")
        
        for step in tqdm.tqdm(range(self.cfg.max_steps)):
            self.step = step
            
            # Random sample from dataset
            data_start = time.time()
            idx = np.random.randint(0, len(self.dataset))
            batch = self.dataset[idx]
            data_time = time.time() - data_start
            
            # Move to device
            device_start = time.time()
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            device_time = time.time() - device_start
            
            # Training step
            optim_start = time.time()
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            
            stats = self.train_step(batch)
            
            # Backward pass
            backward_start = time.time()
            stats["loss"] = torch.tensor(stats["loss"], requires_grad=True)
            stats["loss"].backward()
            backward_time = time.time() - backward_start
            
            # Optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
            optim_time = time.time() - optim_start
            
            # Add overall timing info
            stats["time_data"] = data_time
            stats["time_device"] = device_time
            stats["time_backward"] = backward_time
            stats["time_optim"] = optim_time
            
            # Logging
            if step % 10 == 0:  # More frequent logging for timing analysis
                self.writer.add_scalar("train/loss", stats["loss"], step)
                self.writer.add_scalar("train/psnr", stats["psnr"], step)
                self.writer.add_scalar("train/n_gaussians", stats["n_gaussians"], step)
                
                # Log timing info
                self.writer.add_scalar("timing/data", stats["time_data"], step)
                self.writer.add_scalar("timing/gaussian", stats["time_gaussian"], step)
                self.writer.add_scalar("timing/render", stats["time_render"], step)
                self.writer.add_scalar("timing/backward", stats["time_backward"], step)
                self.writer.add_scalar("timing/total", stats["time_total"] + stats["time_optim"], step)
                
                print(f"Step {step}: Loss={stats['loss']:.4f}, PSNR={stats['psnr']:.2f}, "
                      f"Gaussians={stats['n_gaussians']}")
                print(f"  Timing - Data:{stats['time_data']:.3f}s, Gaussian:{stats['time_gaussian']:.3f}s, "
                      f"Render:{stats['time_render']:.3f}s, Backward:{stats['time_backward']:.3f}s, "
                      f"Total:{stats['time_total'] + stats['time_optim']:.3f}s")
            
            # Evaluation
            if step in self.cfg.eval_steps or (step > 0 and step % 1000 == 0):
                self.evaluate(step)
            
            # Save checkpoint
            if step in self.cfg.save_steps:
                self.save_checkpoint(step)
    
    def evaluate(self, step: int):
        """評価の実行"""
        print(f"Evaluating at step {step}...")
        
        metrics = {"psnr": [], "ssim": [], "n_gaussians": []}
        
        with torch.no_grad():
            for i in range(min(10, len(self.val_dataset))):  # Evaluate on first 10 validation frames
                batch = self.val_dataset[i]
                
                # Move to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Get combined gaussians
                combined_splats = self.dynamic_system.get_combined_gaussians(frame_data=batch)
                
                # Render
                camtoworld = batch["camtoworld"].unsqueeze(0)
                K = batch["K"].unsqueeze(0)
                image_gt = batch["image"].unsqueeze(0) / 255.0
                
                # Resize if needed
                H, W = image_gt.shape[1:3]
                if H != self.cfg.render_size or W != self.cfg.render_size:
                    image_gt = F.interpolate(
                        image_gt.permute(0, 3, 1, 2),
                        size=(self.cfg.render_size, self.cfg.render_size),
                        mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    K = K.clone()
                    K[:, 0, 2] *= self.cfg.render_size / W
                    K[:, 1, 2] *= self.cfg.render_size / H
                    K[:, 0, 0] *= self.cfg.render_size / W
                    K[:, 1, 1] *= self.cfg.render_size / H
                
                renders, alphas, info = rasterization(
                    means=combined_splats["means"],
                    quats=F.normalize(combined_splats["quats"], dim=-1),
                    scales=torch.exp(combined_splats["scales"]),
                    opacities=torch.sigmoid(combined_splats["opacities"]),
                    colors=combined_splats["sh0"],
                    viewmats=torch.linalg.inv(camtoworld),
                    Ks=K,
                    width=self.cfg.render_size,
                    height=self.cfg.render_size,
                    sh_degree=self.cfg.sh_degree,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                )
                
                # Metrics
                psnr_val = self.psnr(renders, image_gt).item()
                
                # TODO: SSIM計算を一時的にマスク - padding size error回避のため
                # ssim_val = self.ssim(renders, image_gt).item()
                ssim_val = 0.0  # 一時的にダミー値
                
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["n_gaussians"].append(len(combined_splats["means"]))
                
                # Save first render
                if i == 0:
                    render_np = renders.squeeze(0).cpu().numpy()
                    gt_np = image_gt.squeeze(0).cpu().numpy()
                    
                    # Save side-by-side comparison
                    comparison = np.hstack([gt_np, render_np])
                    imageio.imwrite(
                        f"{self.render_dir}/val_step{step:06d}.png",
                        (comparison * 255).astype(np.uint8)
                    )
        
        # Log metrics
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        avg_gaussians = np.mean(metrics["n_gaussians"])
        
        self.writer.add_scalar("val/psnr", avg_psnr, step)
        self.writer.add_scalar("val/ssim", avg_ssim, step)
        self.writer.add_scalar("val/n_gaussians", avg_gaussians, step)
        
        print(f"Validation: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f}, Gaussians={avg_gaussians:.0f}")
    
    def save_checkpoint(self, step: int):
        """チェックポイントの保存"""
        checkpoint = {
            "step": step,
            "dynamic_system": self.dynamic_system.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "config": self.cfg,
        }
        
        torch.save(checkpoint, f"{self.ckpt_dir}/ckpt_{step:06d}_rank{self.world_rank}.pt")
        print(f"✓ Checkpoint saved: step {step}")


def main():
    """メイン関数"""
    
    # Parse config
    cfg = tyro.cli(DynamicConfig)
    
    # Create result directory
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    # Save config
    with open(f"{cfg.result_dir}/config.yaml", "w") as f:
        yaml.dump(cfg.__dict__, f, default_flow_style=False)
    
    # Single GPU training for now
    local_rank = 0
    world_rank = 0
    world_size = 1
    
    # Run training
    runner = DynamicRunner(local_rank, world_rank, world_size, cfg)
    runner.train()
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()