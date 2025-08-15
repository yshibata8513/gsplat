#!/usr/bin/env python3
"""
Full Point Cloud Dynamic Render Test

全点群・全動的物体を使用した深度色付きレンダリングテスト
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('/workspace/gsplat/examples')

from datasets.pandaset import PandaSetDataset
from models.dynamic_splats import create_dynamic_splats_from_points
from gsplat.rendering import rasterization

# utils.pyを直接インポート
import importlib.util
utils_spec = importlib.util.spec_from_file_location("core_utils", "/workspace/gsplat/examples/utils.py")
core_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(core_utils)

def depth_colorize(depth, min_depth=None, max_depth=None):
    """深度を色付きで可視化"""
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
    
    # 深度を0-1に正規化
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # Turboカラーマップを使用（距離に応じて青→緑→黄→赤）
    colored = plt.cm.turbo(depth_norm)[:, :, :3]  # RGBのみ取得
    
    return colored

def full_pointcloud_render():
    """全点群を使用した深度色付きレンダリング"""
    print("=" * 60)
    print("Full Point Cloud Dynamic Render Test")
    print("=" * 60)
    
    # 出力ディレクトリ
    os.makedirs("results/full_render", exist_ok=True)
    
    # データセット
    dataset = PandaSetDataset(
        "/workspace/gsplat/extracted_data_new", 
        split="train", 
        test_every=10
    )
    
    print(f"Dataset: {len(dataset)} frames")
    
    # 全静的点群を使用（メモリ制約で一部をサンプリング）
    # 真ん中くらいのインデックス（時刻）を使用。以降一貫してこのデータのみでレンダリング結果の検証を行う。
    middle_idx = len(dataset) // 2
    sample = dataset[middle_idx]
    total_static = len(sample["static_points"])
    sample_time = sample["time"]  # この時刻を保存
    print(f"Using middle frame {middle_idx} at time {sample_time:.3f}")

    
    # メモリ制約を考慮して300K点をサンプリング
    static_sample_rate = min(1.0, 300000 / total_static)
    static_indices = np.random.choice(total_static, 
                                    int(total_static * static_sample_rate), 
                                    replace=False)
    static_points = sample["static_points"][static_indices].numpy()
    
    print(f"Static points: {len(static_points):,} (sampled from {total_static:,})")
    
    # 全動的物体を含める（大きすぎる場合は制限）
    selected_actors = {}
    total_dynamic = 0
    
    for actor_id, points in sample["dynamic_points"].items():
        # 各アクターを制限（メモリ効率のため）
        max_points_per_actor = 2000
        if len(points) > max_points_per_actor:
            indices = np.random.choice(len(points), max_points_per_actor, replace=False)
            selected_actors[actor_id] = points[indices].numpy()
        else:
            selected_actors[actor_id] = points.numpy()
        total_dynamic += len(selected_actors[actor_id])
    
    print(f"Dynamic actors: {len(selected_actors)}")
    print(f"Dynamic points: {total_dynamic:,}")
    print(f"Total points: {len(static_points) + total_dynamic:,}")
    
    # アクター姿勢データ収集（sample時刻前後のデータ）
    actor_poses = {}
    for actor_id in selected_actors.keys():
        times, poses = [], []
        
        # dataset.actor_posesから直接アクセスして時刻でフィルタリング
        if actor_id in dataset.actor_poses:
            actor_data = dataset.actor_poses[actor_id]
            all_times = np.array(actor_data["times"])
            all_matrices = actor_data["matrices"]
            
            # まず最も時間が近い2つを取得
            time_diffs = np.abs(all_times - sample_time)
            closest_indices = np.argsort(time_diffs)[:min(2, len(all_times))]
            
            # 過去と未来のデータがあるかチェック
            closest_times = all_times[closest_indices]
            has_past = np.any(closest_times < sample_time)
            has_future = np.any(closest_times > sample_time)
            
            # 最も近い2つを追加
            for idx in closest_indices:
                times.append(all_times[idx])
                poses.append(all_matrices[idx].astype(np.float32))
            
            # 過去のデータがなければ追加
            if not has_past:
                past_mask = all_times < sample_time
                if np.any(past_mask):
                    past_times = all_times[past_mask]
                    past_idx = np.argmax(past_times)  # 最も近い過去のデータ
                    actual_past_idx = np.where(past_mask)[0][past_idx]
                    times.append(all_times[actual_past_idx])
                    poses.append(all_matrices[actual_past_idx].astype(np.float32))
            
            # 未来のデータがなければ追加
            if not has_future:
                future_mask = all_times > sample_time
                if np.any(future_mask):
                    future_times = all_times[future_mask]
                    future_idx = np.argmin(future_times)  # 最も近い未来のデータ
                    actual_future_idx = np.where(future_mask)[0][future_idx]
                    times.append(all_times[actual_future_idx])
                    poses.append(all_matrices[actual_future_idx].astype(np.float32))
            
            # 時刻順にソート
            if times:
                sorted_indices = np.argsort(times)
                times = [times[i] for i in sorted_indices]
                poses = [poses[i] for i in sorted_indices]
        
        if times:
            actor_poses[actor_id] = {"times": times, "poses": poses}
            print(f"  Actor {actor_id}: {len(times)} poses at times {[f'{t:.3f}' for t in times]}")
        else:
            actor_poses[actor_id] = {"times": [sample_time], "poses": [np.eye(4, dtype=np.float32)]}
            print(f"  Actor {actor_id}: No pose data, using identity")
    
    # この時点でsampleのカメラパラメータを使って点群を深度で色分け、動的物体は白
    print("\nApplying depth-based coloring to point clouds...")
    
    # カメラパラメータを取得
    camtoworld = sample["camtoworld"].numpy()
    K = sample["K"].numpy()
    
    # カメラ座標系への変換行列
    worldtocam = np.linalg.inv(camtoworld)
    
    # 静的点群をカメラ座標系に変換して深度を計算
    static_points_cam = (worldtocam[:3, :3] @ static_points[:, :3].T + worldtocam[:3, 3:4]).T
    static_depths = static_points_cam[:, 2]  # Z座標が深度
    
    # 深度に基づいて色を設定（turboカラーマップ）
    depth_min, depth_max = 2.0, 50.0
    static_depths_norm = np.clip((static_depths - depth_min) / (depth_max - depth_min), 0, 1)
    
    # Turboカラーマップを使用
    static_colors = plt.cm.turbo(static_depths_norm)[:, :3]  # RGBのみ
    static_points[:, 3:6] = static_colors
    
    # 動的物体は白色に設定
    for actor_id, points in selected_actors.items():
        selected_actors[actor_id][:, 3:6] = 1.0  # 白色
    
    print(f"✓ Applied depth-based colors (depth range: {depth_min:.1f}m - {depth_max:.1f}m)")
    print(f"  Static: Turbo colormap, Dynamic: White")
    

    # 初期レンダリングを実行して色付け確認
    print("\nPerforming initial rendering with colored point clouds...")
    
    # 初期化パラメータ（色はすでに設定済み）
    init_params = {
        "init_scale": 0.3,      # 小さなスケール
        "init_opa": 0.8,        # 高い不透明度
        "sh_degree": 0,         # 色はすでに設定済みのでSH=0
    }
    
    print("Creating full dynamic gaussian system...")
    dynamic_splats = create_dynamic_splats_from_points(
        static_points=static_points,
        dynamic_points=selected_actors,
        actor_poses=actor_poses,
        init_params=init_params,
        device="cuda"
    )
    
    print(f"✓ System created:")
    print(f"  Static gaussians: {dynamic_splats.get_static_count()}")
    print(f"  Dynamic gaussians: {dynamic_splats.get_dynamic_count()}")
    print(f"  Total: {dynamic_splats.get_total_count()}")

    # 初期レンダリング結果の可視化
    print("\nVisualizing initial colored point cloud rendering...")
    
    # dynamic_splats.get_combined_gaussiansでデータを取り出し、rasterizationを使ってrender
    with torch.no_grad():
        # 統合ガウシアンを取得
        combined = dynamic_splats.get_combined_gaussians(float(sample_time))
        
        # カメラパラメータをGPUに転送
        camtoworld_cuda = sample["camtoworld"].unsqueeze(0).cuda()
        K_cuda = sample["K"].unsqueeze(0).cuda()
        target = sample["image"].unsqueeze(0).cuda() / 255.0
        
        # レンダリングサイズ
        render_size = 512
        H, W = target.shape[1:3]
        
        # カメラパラメータ調整
        K_resized = K_cuda.clone()
        K_resized[:, 0, 2] *= render_size / W  # cx
        K_resized[:, 1, 2] *= render_size / H  # cy
        K_resized[:, 0, 0] *= render_size / W  # fx
        K_resized[:, 1, 1] *= render_size / H  # fy
        
        # レンダリング
        renders, alphas, info = rasterization(
            means=combined["means"],
            quats=F.normalize(combined["quats"], dim=-1),
            scales=torch.exp(combined["scales"]),
            opacities=torch.sigmoid(combined["opacities"]),
            colors=combined["sh0"],  # 色はすでに設定済み
            viewmats=torch.linalg.inv(camtoworld_cuda),
            Ks=K_resized,
            width=render_size,
            height=render_size,
            sh_degree=0,
            near_plane=0.1,
            far_plane=200.0,
        )
        
        # 結果をCPUに転送して保存
        initial_render = renders.squeeze(0).cpu().numpy()
        initial_alpha = alphas.squeeze(0).cpu().numpy()
        
        # Ground Truthをリサイズ
        target_resized = F.interpolate(
            target.permute(0, 3, 1, 2),
            size=(render_size, render_size),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    
    # 初期レンダリング結果を保存
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(target_resized)
    axes[0].set_title(f"Ground Truth (Frame {middle_idx})")
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(initial_render, 0, 1))
    axes[1].set_title(f"Initial Colored Render\n(Static: Depth-colored, Dynamic: White)")
    axes[1].axis('off')
    
    axes[2].imshow(initial_alpha, cmap='viridis')
    axes[2].set_title(f"Alpha Channel\n(Mean: {initial_alpha.mean():.3f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/full_render/initial_colored_render.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Initial rendering saved to results/full_render/initial_colored_render.png")
    print(f"  Rendered gaussians: {info.get('n_render', 'unknown')}")
    print(f"  RGB range: [{initial_render.min():.3f}, {initial_render.max():.3f}]")
    
    # 以降の処理ではset_depth_colorsを使わない（色はすでに設定済み）
    # set_depth_colors(dynamic_splats)  # コメントアウト
    
    # 高解像度でレンダリング
    render_size = 512
    
    # middle_idx前後しかposeを登録していないので、middle_idxのみレンダリング
    frames_to_render = [middle_idx]
    results = []
    
    for i, frame_idx in enumerate(frames_to_render):
        if frame_idx >= len(dataset):
            frame_idx = len(dataset) - 1
            
        sample = dataset[frame_idx]
        time_val = sample["time"]
        camtoworld = sample["camtoworld"].unsqueeze(0).cuda()
        K = sample["K"].unsqueeze(0).cuda()
        target = sample["image"].unsqueeze(0).cuda() / 255.0
        
        print(f"\nRendering frame {frame_idx} (time: {time_val:.3f})...")
        
        # リサイズ
        H, W = target.shape[1:3]
        target_resized = F.interpolate(
            target.permute(0, 3, 1, 2),
            size=(render_size, render_size),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        # カメラパラメータ調整
        K_resized = K.clone()
        K_resized[:, 0, 2] *= render_size / W  # cx
        K_resized[:, 1, 2] *= render_size / H  # cy
        K_resized[:, 0, 0] *= render_size / W  # fx
        K_resized[:, 1, 1] *= render_size / H  # fy
        
        # 統合レンダリング（深度付き）
        with torch.no_grad():
            combined = dynamic_splats.get_combined_gaussians(float(time_val))
            
            # 色はすでに設定済み（静的: 深度色、動的: 白）
            colors = combined["sh0"]
            
            renders, alphas, info = rasterization(
                means=combined["means"],
                quats=F.normalize(combined["quats"], dim=-1),
                scales=torch.exp(combined["scales"]),
                opacities=torch.sigmoid(combined["opacities"]),
                colors=colors,
                viewmats=torch.linalg.inv(camtoworld),
                Ks=K_resized,
                width=render_size,
                height=render_size,
                sh_degree=0,
                near_plane=0.1,
                far_plane=200.0,
            )
            
            # 深度は座標から計算
            rgb_render = renders
            
            # カメラから各ピクセルまでの深度を計算
            means_cam = torch.matmul(combined["means"], torch.linalg.inv(camtoworld).squeeze(0)[:3, :3].T) + torch.linalg.inv(camtoworld).squeeze(0)[:3, 3]
            depths = means_cam[:, 2]  # Z座標が深度
            
            # レンダリング重みに基づいて深度マップを作成（簡易版）
            depth_map = torch.zeros((render_size, render_size), device=combined["means"].device)
            alpha_2d = alphas.squeeze(0).squeeze(-1)
            
            # 平均深度を使用
            mean_depth = depths.mean().item()
            depth_render = torch.full((render_size, render_size), mean_depth, device=combined["means"].device).unsqueeze(-1)
        
        # 結果を保存
        target_np = target_resized.squeeze(0).cpu().numpy()
        rgb_np = rgb_render.squeeze(0).cpu().numpy()
        alpha_np = alphas.squeeze(0).cpu().numpy()
        
        # 深度を色付きで表示
        depth_np = depth_render.squeeze(-1).cpu().numpy()  # [H, W]
        depth_colored = depth_colorize(depth_np, min_depth=2.0, max_depth=50.0)
        
        # クリップして正規化
        target_np = np.clip(target_np, 0, 1)
        rgb_np = np.clip(rgb_np, 0, 1)
        alpha_np = np.clip(alpha_np, 0, 1)
        
        results.append({
            'frame': frame_idx,
            'time': time_val,
            'target': target_np,
            'render': rgb_np,
            'depth': depth_colored,
            'alpha': alpha_np,
            'n_gaussians': dynamic_splats.get_total_count()
        })
        
        print(f"  Rendered gaussians: {info.get('n_render', 'unknown')}")
        print(f"  RGB range: [{rgb_np.min():.3f}, {rgb_np.max():.3f}]")
        print(f"  Alpha range: [{alpha_np.min():.3f}, {alpha_np.max():.3f}]")
        print(f"  Mean alpha: {alpha_np.mean():.3f}")
    
    # 結果画像を作成
    create_full_comparison(results, "results/full_render")
    
    print("✅ Full point cloud render test completed!")
    print("Check results/full_render/ for output images")

def set_depth_colors_unused(dynamic_splats):
    """深度に基づいて色を設定（未使用）"""
    print("Setting depth-based colors...")
    
    with torch.no_grad():
        # 静的ガウシアンの深度色設定
        static_means = dynamic_splats.static_splats["means"]
        static_depths = static_means[:, 2]  # Z座標を深度として使用
        
        # 深度を正規化 (2m-50mの範囲)
        depth_min, depth_max = 2.0, 50.0
        normalized_depths = torch.clamp((static_depths - depth_min) / (depth_max - depth_min), 0, 1)
        
        # Turboカラーマップを近似（赤→黄→緑→青）
        colors = torch.zeros_like(static_means)  # [N, 3]
        
        # 簡単な色マッピング: 近い=暖色、遠い=寒色
        colors[:, 0] = 1.0 - normalized_depths  # 赤: 近いほど強く
        colors[:, 1] = torch.sin(normalized_depths * np.pi)  # 緑: 中間で強く
        colors[:, 2] = normalized_depths  # 青: 遠いほど強く
        
        # 球面調和関数の0次項として設定
        C0 = 0.28209479177387814
        sh_colors = (colors - 0.5) / C0
        dynamic_splats.static_splats["sh0"].data = sh_colors.unsqueeze(1)
        
        # 動的ガウシアンも同様に設定
        for actor_key, splats in dynamic_splats.dynamic_splats.items():
            dynamic_means = splats["means"]
            dynamic_depths = dynamic_means[:, 2]
            normalized_depths = torch.clamp((dynamic_depths - depth_min) / (depth_max - depth_min), 0, 1)
            
            colors = torch.zeros_like(dynamic_means)
            colors[:, 0] = 1.0 - normalized_depths + 0.2  # 動的物体は少し赤みを強く
            colors[:, 1] = torch.sin(normalized_depths * np.pi)
            colors[:, 2] = normalized_depths
            
            sh_colors = (colors - 0.5) / C0
            splats["sh0"].data = sh_colors.unsqueeze(1)
    
    print("✓ Depth colors set")

def create_full_comparison(results, output_dir):
    """完全な比較画像を作成"""
    n_frames = len(results)
    
    # 大きな比較画像
    fig, axes = plt.subplots(4, n_frames, figsize=(8*n_frames, 20))
    if n_frames == 1:
        axes = axes.reshape(4, 1)
    
    for i, result in enumerate(results):
        # Ground Truth
        axes[0, i].imshow(result['target'])
        axes[0, i].set_title(f"Ground Truth\nFrame {result['frame']}, t={result['time']:.3f}s", fontsize=14)
        axes[0, i].axis('off')
        
        # Depth-colored render
        axes[1, i].imshow(result['render'])
        axes[1, i].set_title(f"Depth-Colored Gaussians\n{result['n_gaussians']:,} total", fontsize=14)
        axes[1, i].axis('off')
        
        # Depth visualization
        axes[2, i].imshow(result['depth'])
        axes[2, i].set_title("Depth Map\n(Near=Red, Far=Blue)", fontsize=14)
        axes[2, i].axis('off')
        
        # Alpha channel
        axes[3, i].imshow(result['alpha'], cmap='viridis')
        axes[3, i].set_title(f"Opacity\nMean: {result['alpha'].mean():.3f}", fontsize=14)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/full_comparison.png", 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 個別の詳細画像も保存
    for i, result in enumerate(results):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(result['target'])
        axes[0, 0].set_title(f"Ground Truth - Frame {result['frame']}", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(result['render'])
        axes[0, 1].set_title(f"Depth-Colored Point Cloud\n{result['n_gaussians']:,} Gaussians", fontsize=12)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(result['depth'])
        axes[1, 0].set_title("Depth Visualization", fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result['alpha'], cmap='viridis')
        axes[1, 1].set_title("Opacity Map", fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_frame_{result['frame']:03d}.png", 
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Saved comparison images to {output_dir}")

if __name__ == "__main__":
    full_pointcloud_render()