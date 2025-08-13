

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH3DのLiDARスキャンをscan_alignment.mlpの変換行列で統合し、
（任意で）視錐台クリップ・外れ値除去・ボクセル間引きまで行って
gsplat初期化に使えるサイズに整えるスクリプト。

想定ワークフロー:
  A) (任意) per-scan粗ボクセル間引き --pre_voxel
  B) MLPのMLMatrix44で各PLYを世界座標に変換して統合
  C) COLMAPの cameras/images から視錐台クリップ (--sparse_dir)
  D) (任意) 統計的外れ値除去 (--stat_nb > 0)
  E) ボクセル間引き (--target_points もしくは --voxel_size)

出力: merged_lidar_*.ply （Binary）
"""

import argparse
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import open3d as o3d


# ------------------------
# MLP (MeshLab Project) 読み取り
# ------------------------
def parse_mlp_file(mlp_path: Path) -> Dict[str, np.ndarray]:
    tree = ET.parse(str(mlp_path))
    root = tree.getroot()
    transformations = {}
    for mesh in root.findall(".//MLMesh"):
        filename = mesh.get("filename")
        mat = mesh.find("MLMatrix44")
        if filename is None or mat is None or mat.text is None:
            continue
        vals = [float(x) for x in mat.text.strip().split()]
        if len(vals) != 16:
            continue
        M = np.array(vals, dtype=np.float64).reshape(4, 4)
        transformations[filename] = M
        print(f"[MLP] {filename}: found 4x4 transform")
    return transformations


# ------------------------
# COLMAP 読み取り (txt / bin)
#   - cameras: fx, fy, cx, cy, width, height
#   - images:  R(w2c), t
# ------------------------
def _qvec2rotmat(qw, qx, qy, qz):
    # COLMAP: images.txt の四元数は (qw, qx, qy, qz), world->cam
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = q / n
    R = np.array([
        [1-2*qy*qy-2*qz*qz,   2*qx*qy-2*qz*qw,     2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,     1-2*qx*qx-2*qz*qz,   2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,     2*qy*qz+2*qx*qw,     1-2*qx*qx-2*qy*qy]
    ], dtype=np.float64)
    return R


def read_cameras_txt(path: Path) -> Dict[int, dict]:
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        cam_id = int(toks[0])
        model = toks[1]
        width = int(toks[2]); height = int(toks[3])
        params = list(map(float, toks[4:]))
        if model == "PINHOLE" or model == "OPENCV" or model == "OPENCV_FISHEYE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE" or model == "SIMPLE_RADIAL" or model == "RADIAL":
            f, cx, cy = params[0], params[1], params[2]
            fx, fy = f, f
        else:
            # 未対応モデルはPINHOLE相当に落とす
            f, cx, cy = params[0], params[1], params[2]
            fx, fy = f, f
        cams[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=width, h=height, model=model)
    print(f"[COLMAP] cameras.txt: {len(cams)} cameras")
    return cams


def read_images_txt(path: Path) -> List[dict]:
    images = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 10:
            continue
        img_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = toks[9]
        R = _qvec2rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        images.append(dict(id=img_id, cam_id=cam_id, R=R, t=t, name=name))
        # 次行は2D点列。スキップ。
        if i < len(lines) and not lines[i].startswith("#"):
            i += 1
    print(f"[COLMAP] images.txt: {len(images)} images (poses)")
    return images


# --- .bin を簡易読み込み（最小限: PINHOLE系&undistort前提） ---
def read_cameras_bin(path: Path) -> Dict[int, dict]:
    cams = {}
    with open(path, "rb") as f:
        little_endian = True
        def read_next(fmt):
            return struct.unpack(("<" if little_endian else ">") + fmt, f.read(struct.calcsize(fmt)))
        num_cams, = read_next("Q")
        for _ in range(num_cams):
            cam_id, model_id, width, height = read_next("iiQQ")
            # PARAMS数（model_idで分岐）
            # 0:SIMPLE_PINHOLE,1:PINHOLE,2:SIMPLE_RADIAL,3:RADIAL,4:OPENCV,5:OPENCV_FISHEYE など
            num_params = {0:3,1:4,2:4,3:5,4:8,5:4}.get(model_id, 4)
            params = np.array(read_next("d"*num_params))
            if model_id in (1,4,5):  # PINHOLE/OPENCV/FISHEYE
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            cams[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=int(width), h=int(height), model=str(model_id))
    print(f"[COLMAP] cameras.bin: {len(cams)} cameras")
    return cams


def read_images_bin(path: Path) -> List[dict]:
    images = []
    with open(path, "rb") as f:
        little_endian = True
        def read_next(fmt):
            return struct.unpack(("<" if little_endian else ">") + fmt, f.read(struct.calcsize(fmt)))
        num_imgs, = read_next("Q")
        for _ in range(num_imgs):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = read_next("idddddddi")
            name_len, = read_next("Q")
            name = f.read(name_len).decode("utf-8")
            R = _qvec2rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            # 2D点群は読み飛ばし
            num_points2D, = read_next("Q")
            f.seek(num_points2D * (2*8 + 8 + 8), 1)  # x,y(double), point3D_id(long), dummy?
            images.append(dict(id=img_id, cam_id=cam_id, R=R, t=t, name=name))
    print(f"[COLMAP] images.bin: {len(images)} images (poses)")
    return images


def load_colmap_sparse(sparse_dir: Path) -> Tuple[Dict[int, dict], List[dict]]:
    cams_txt = sparse_dir / "cameras.txt"
    imgs_txt = sparse_dir / "images.txt"
    if cams_txt.exists() and imgs_txt.exists():
        cams = read_cameras_txt(cams_txt)
        imgs = read_images_txt(imgs_txt)
        return cams, imgs
    cams_bin = sparse_dir / "cameras.bin"
    imgs_bin = sparse_dir / "images.bin"
    if cams_bin.exists() and imgs_bin.exists():
        cams = read_cameras_bin(cams_bin)
        imgs = read_images_bin(imgs_bin)
        return cams, imgs
    raise FileNotFoundError(f"No COLMAP cameras/images found under {sparse_dir}")


# ------------------------
# 変換＆統合
# ------------------------
def load_and_transform_ply(ply_path: Path, M: np.ndarray, pre_voxel: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return pcd
    if pre_voxel > 0:
        pcd = pcd.voxel_down_sample(pre_voxel)
    pcd.transform(M)
    # 色が無ければ仮のグレー(0.8)
    if len(pcd.colors) == 0:
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.8, dtype=np.float64))
    return pcd


# ------------------------
# 視錐台クリップ（少なくとも N 枚に映る点のみ残す）
#   - バッチ処理で巨大点群にも対応
# ------------------------
def frustum_cull(points_xyz: np.ndarray,
                 cams: Dict[int, dict],
                 imgs: List[dict],
                 visible_in: int = 1,
                 margin_px: int = 0,
                 batch_size: int = 2_000_000) -> np.ndarray:
    N = points_xyz.shape[0]
    keep = np.zeros(N, dtype=bool)
    idxs = np.arange(N)

    # 準備：各画像のKとw2c
    Ks, Rs, ts, sizes = [], [], [], []
    for im in imgs:
        c = cams[im["cam_id"]]
        fx, fy, cx, cy = c["fx"], c["fy"], c["cx"], c["cy"]
        w, h = c["w"], c["h"]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        Ks.append(K); Rs.append(im["R"]); ts.append(im["t"]); sizes.append((w, h))

    Ks  = np.stack(Ks)   # [M,3,3]
    Rs  = np.stack(Rs)   # [M,3,3]
    ts  = np.stack(ts)   # [M,3]
    Ws  = np.array([wh[0] for wh in sizes])
    Hs  = np.array([wh[1] for wh in sizes])
    M = Ks.shape[0]

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        P = points_xyz[s:e]                      # [B,3]
        # [M,3,B] = [M,3,3]@[3,B] + [M,3,1]
        Pw = P.T[np.newaxis, ...]                # [1,3,B]
        Pc = (Rs @ Pw) + ts[..., np.newaxis]     # [M,3,B]
        Z = Pc[:, 2, :]                          # [M,B]
        X = Pc[:, 0, :]; Y = Pc[:, 1, :]
        # 正面のみに
        in_front = Z > 1e-6
        # 画像へ投影
        u = (Ks[:, 0, 0:1] * X / Z) + Ks[:, 0, 2:3]   # [M,B,1]
        v = (Ks[:, 1, 1:2] * Y / Z) + Ks[:, 1, 2:3]
        u = u.squeeze(-1); v = v.squeeze(-1)          # [M,B]

        # 画面内（マージンあり）
        cond_u = (u >= -margin_px) & (u < (Ws[:, None] + margin_px))
        cond_v = (v >= -margin_px) & (v < (Hs[:, None] + margin_px))
        vis = in_front & cond_u & cond_v              # [M,B]
        vis_count = np.count_nonzero(vis, axis=0)     # [B]
        keep[s:e] = keep[s:e] | (vis_count >= visible_in)
        print(f"[Frustum] batch {s//batch_size+1}: keep {np.count_nonzero(keep[s:e])}/{e-s}")

    kept = idxs[keep]
    print(f"[Frustum] total kept: {kept.size}/{N} ({kept.size/N*100:.2f}%)")
    return keep


# ------------------------
# 目標点数から voxel_size を推定
# ------------------------
def estimate_voxel_for_target(pcd: o3d.geometry.PointCloud, target: int) -> float:
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.array(bbox.get_extent(), dtype=np.float64)
    vol = float(extent[0] * extent[1] * extent[2]) if np.all(extent > 0) else 1.0
    v = (vol / max(target, 1)) ** (1.0/3.0)
    return float(v)


# ------------------------
# メイン
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Merge ETH3D LiDAR scans -> (optional) frustum cull -> outlier removal -> voxel downsample")
    ap.add_argument("--scan_dir", default="scan_clean", help="Directory containing scan PLY files")
    ap.add_argument("--mlp_file", default="scan_clean/scan_alignment.mlp", help="Path to scan_alignment.mlp file")
    ap.add_argument("--sparse_dir", default="sparse/0", help="Path to COLMAP sparse dir (contains cameras/images .txt or .bin)")
    ap.add_argument("--output", default="merged_lidar.ply", help="Output merged PLY")
    # 前処理
    ap.add_argument("--pre_voxel", type=float, default=0.0, help="Per-scan pre-voxel size (meters). 0=disable")
    # 視錐台
    ap.add_argument("--enable_frustum", action="store_true", help="Enable frustum culling using COLMAP poses")
    ap.add_argument("--visible_in", type=int, default=1, help="Keep points visible in at least N images")
    ap.add_argument("--margin_px", type=int, default=0, help="Pixel margin around image bounds")
    ap.add_argument("--batch_size", type=int, default=2_000_000, help="Batch size for frustum culling")
    # 外れ値
    ap.add_argument("--stat_nb", type=int, default=0, help="Statistical outlier removal neighbors. 0=disable")
    ap.add_argument("--stat_std", type=float, default=2.0, help="Std ratio for outlier removal")
    # ボクセル
    ap.add_argument("--voxel_size", type=float, default=0.0, help="Final voxel size (meters). If >0, used directly.")
    ap.add_argument("--target_points", type=int, default=100_000, help="Target number of points if voxel_size==0")
    # 保存
    ap.add_argument("--ascii", action="store_true", help="Write ASCII PLY (default binary)")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    mlp_file = Path(args.mlp_file)
    sparse_dir = Path(args.sparse_dir)
    output = Path(args.output)

    print(f"== Merge params ==")
    print(f"scan_dir={scan_dir}")
    print(f"mlp_file={mlp_file}")
    print(f"sparse_dir={sparse_dir}  (used only if --enable_frustum)")
    print(f"pre_voxel={args.pre_voxel}  enable_frustum={args.enable_frustum}  visible_in={args.visible_in}")
    print(f"stat_nb={args.stat_nb}  stat_std={args.stat_std}")
    print(f"voxel_size={args.voxel_size}  target_points={args.target_points}")
    print("===============")

    transforms = parse_mlp_file(mlp_file)
    if not transforms:
        raise RuntimeError("No transforms found in MLP")

    # 1) per-scan 読み込み＋変換（＋pre-voxel）
    merged = o3d.geometry.PointCloud()
    total_before = 0
    for fname, M in transforms.items():
        ply_path = scan_dir / fname
        if not ply_path.exists():
            print(f"[WARN] {ply_path} missing, skip")
            continue
        p = load_and_transform_ply(ply_path, M, pre_voxel=args.pre_voxel)
        n = len(p.points)
        if n == 0:
            continue
        total_before += n
        merged += p
        print(f"[Merge] {fname}: {n} pts (accum={len(merged.points)})")
    if len(merged.points) == 0:
        raise RuntimeError("No points loaded after merging")

    print(f"[Merge] total points after merge: {len(merged.points)} (before any culling)")
    # 2) 視錐台クリップ
    if args.enable_frustum:
        cams, imgs = load_colmap_sparse(sparse_dir)
        P = np.asarray(merged.points)
        keep = frustum_cull(P, cams, imgs, visible_in=args.visible_in,
                            margin_px=args.margin_px, batch_size=args.batch_size)
        merged = merged.select_by_index(np.flatnonzero(keep))
        print(f"[Frustum] after culling: {len(merged.points)} pts")

    # 3) 外れ値除去（軽く1回）
    if args.stat_nb and args.stat_nb > 0:
        merged, ind = merged.remove_statistical_outlier(nb_neighbors=args.stat_nb, std_ratio=args.stat_std)
        print(f"[Outlier] after statistical removal: {len(merged.points)} pts")

    # 4) ボクセル（ターゲット点数 or 指定ボクセル）
    if args.voxel_size > 0:
        v = float(args.voxel_size)
    else:
        v = estimate_voxel_for_target(merged, args.target_points)
    print(f"[Voxel] initial voxel_size = {v:.6f} m")

    def downsample_to(pc: o3d.geometry.PointCloud, target: int, v0: float):
        v = v0
        for _ in range(5):
            ds = pc.voxel_down_sample(v)
            n = len(ds.points)
            print(f"[Voxel] try v={v:.6f} -> {n} pts")
            if target <= 0 or (0.8*target <= n <= 1.2*target):
                return ds, v
            # 誤差に応じて v を調整（多ければ v↑、少なければ v↓）
            if n > 0 and target > 0:
                v *= (n/target) ** (1/3)
            else:
                break
        return ds, v

    merged, v_final = downsample_to(merged, args.target_points if args.voxel_size <= 0 else 0, v)
    print(f"[Voxel] final voxel_size = {v_final:.6f} m, points = {len(merged.points)}")

    # 5) 保存
    ok = o3d.io.write_point_cloud(str(output), merged, write_ascii=bool(args.ascii), print_progress=True)
    if not ok:
        raise RuntimeError("Failed to save output")
    print(f"✅ Saved: {output}  ({len(merged.points)} pts)  colors={len(merged.colors)>0}")
    print("Done.")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# ETH3DのLiDARスキャンをscan_alignment.mlpの変換行列を使って統合するスクリプト
# """

# import numpy as np
# import xml.etree.ElementTree as ET
# import open3d as o3d
# from pathlib import Path
# import argparse


# def parse_mlp_file(mlp_path):
#     """
#     MeshLab Project (.mlp) ファイルから各スキャンの変換行列を読み取る
    
#     Args:
#         mlp_path: .mlpファイルのパス
    
#     Returns:
#         dict: {filename: transform_matrix} の辞書
#     """
#     tree = ET.parse(mlp_path)
#     root = tree.getroot()
    
#     transformations = {}
    
#     for mesh in root.findall('.//MLMesh'):
#         filename = mesh.get('filename')
#         matrix_elem = mesh.find('MLMatrix44')
        
#         if matrix_elem is not None:
#             # 4x4行列のテキストを解析
#             matrix_text = matrix_elem.text.strip()
#             matrix_values = [float(x) for x in matrix_text.split()]
            
#             # 4x4行列に変換
#             transform_matrix = np.array(matrix_values).reshape(4, 4)
#             transformations[filename] = transform_matrix
            
#             print(f"Found transformation for {filename}")
#             print(f"Matrix:\n{transform_matrix}")
    
#     return transformations


# def load_and_transform_ply(ply_path, transform_matrix):
#     """
#     PLYファイルを読み込み、変換行列を適用する
    
#     Args:
#         ply_path: PLYファイルのパス
#         transform_matrix: 4x4変換行列
    
#     Returns:
#         o3d.geometry.PointCloud: 変換後の点群
#     """
#     print(f"Loading {ply_path}...")
    
#     # PLYファイルを読み込み
#     pcd = o3d.io.read_point_cloud(str(ply_path))
    
#     if len(pcd.points) == 0:
#         print(f"Warning: No points loaded from {ply_path}")
#         return pcd
    
#     print(f"  Loaded {len(pcd.points)} points")
    
#     # 変換行列を適用
#     pcd.transform(transform_matrix)
    
#     # 色情報の有無を確認
#     has_colors = len(pcd.colors) > 0
#     print(f"  Has colors: {has_colors}")
    
#     return pcd


# def merge_lidar_scans(scan_dir, mlp_file, output_path):
#     """
#     scan_alignment.mlpに基づいて全LiDARスキャンを統合
    
#     Args:
#         scan_dir: スキャンファイルが格納されているディレクトリ
#         mlp_file: scan_alignment.mlpファイルのパス
#         output_path: 出力PLYファイルのパス
#     """
#     scan_dir = Path(scan_dir)
#     mlp_file = Path(mlp_file)
#     output_path = Path(output_path)
    
#     print(f"Parsing MLP file: {mlp_file}")
#     transformations = parse_mlp_file(mlp_file)
    
#     if not transformations:
#         print("No transformations found in MLP file!")
#         return
    
#     merged_points = []
#     merged_colors = []
#     has_any_colors = False
    
#     for filename, transform_matrix in transformations.items():
#         ply_path = scan_dir / filename
        
#         if not ply_path.exists():
#             print(f"Warning: {ply_path} not found, skipping...")
#             continue
        
#         # PLYを読み込んで変換
#         pcd = load_and_transform_ply(ply_path, transform_matrix)
        
#         if len(pcd.points) == 0:
#             continue
        
#         # 点を追加
#         points = np.asarray(pcd.points)
#         merged_points.append(points)
        
#         # 色情報がある場合は追加
#         if len(pcd.colors) > 0:
#             colors = np.asarray(pcd.colors)
#             merged_colors.append(colors)
#             has_any_colors = True
#         else:
#             # 色がない場合はダミーの白色を追加
#             dummy_colors = np.ones((len(points), 3)) * 0.8
#             merged_colors.append(dummy_colors)
    
#     if not merged_points:
#         print("No valid point clouds found!")
#         return
    
#     # 全点を結合
#     all_points = np.vstack(merged_points)
#     all_colors = np.vstack(merged_colors) if has_any_colors else None
    
#     print(f"Merged {len(all_points)} total points")
    
#     # 結合した点群を作成
#     merged_pcd = o3d.geometry.PointCloud()
#     merged_pcd.points = o3d.utility.Vector3dVector(all_points)
    
#     if all_colors is not None:
#         merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)
#         print("Merged point cloud has colors")
#     else:
#         print("Merged point cloud has no colors")
    
#     # 保存
#     print(f"Saving merged point cloud to: {output_path}")
#     success = o3d.io.write_point_cloud(str(output_path), merged_pcd)
    
#     if success:
#         print("✅ Successfully saved merged LiDAR point cloud!")
#         print(f"Final point count: {len(merged_pcd.points)}")
#         print(f"Has colors: {len(merged_pcd.colors) > 0}")
#     else:
#         print("❌ Failed to save merged point cloud!")


# def main():
#     parser = argparse.ArgumentParser(description="Merge ETH3D LiDAR scans using scan_alignment.mlp")
#     parser.add_argument("--scan_dir", default="scan_clean", 
#                        help="Directory containing scan PLY files")
#     parser.add_argument("--mlp_file", default="scan_clean/scan_alignment.mlp",
#                        help="Path to scan_alignment.mlp file")
#     parser.add_argument("--output", default="merged_lidar.ply",
#                        help="Output merged PLY file")
    
#     args = parser.parse_args()
    
#     merge_lidar_scans(args.scan_dir, args.mlp_file, args.output)


# if __name__ == "__main__":
#     main()