import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs


class SimpleTrainer:
    """ランダムな3Dガウシアンを使って画像をフィッティングする訓練クラス
    
    3Dガウシアンスプラッティング（3DGS）または2Dガウシアンスプラッティング（2DGS）を使用して、
    目標画像を再現するようにガウシアンパラメータを最適化する。
    """

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        # CUDAデバイスを設定
        self.device = torch.device("cuda:0")
        # 目標画像をGPUメモリに移動
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        # カメラの内部パラメータを設定
        # 視野角90度（π/2ラジアン）でピンホールカメラをシミュレート
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        # 焦点距離を計算（ピンホールカメラモデル）
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        # 3Dガウシアンパラメータを初期化
        self._init_gaussians()

    def _init_gaussians(self):
        """3Dガウシアンのパラメータをランダムに初期化"""
        bd = 2

        # 3D空間での各ガウシアンの中心位置をランダムに配置
        # [-1, 1]の範囲で均等分布からサンプリング
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        # 各軸方向のスケール（ガウシアンの広がり）をランダムに設定
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        # RGB色をランダムに設定（0-1の範囲）
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        # クォータニオン（回転）を均等分布でランダムに生成
        # Marsaglia法を使用して単位球面上で均等にサンプリング
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        # 3D回転を表すクォータニオン (x, y, z, w) を生成
        # 各ガウシアンの向きをランダムに設定
        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        # 不透明度を1で初期化（完全に不透明）
        self.opacities = torch.ones((self.num_points), device=self.device)

        # カメラのビュー行列を定義
        # カメラをz軸方向に8単位後ろに配置（原点から距離8の位置）
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],  # z方向の平行移動
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        # 背景色を黒（0, 0, 0）に設定
        self.background = torch.zeros(d, device=self.device)

        # 最適化対象のパラメータに勾配計算を有効にする
        self.means.requires_grad = True      # 位置
        self.scales.requires_grad = True     # スケール
        self.quats.requires_grad = True      # 回転
        self.rgbs.requires_grad = True       # 色
        self.opacities.requires_grad = True  # 不透明度
        self.viewmat.requires_grad = False   # カメラは固定

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        # Adamオプティマイザで全ての学習可能パラメータを最適化
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        # 平均二乗誤差損失関数（レンダリング画像と目標画像の差を測定）
        mse_loss = torch.nn.MSELoss()
        frames = []  # アニメーション用のフレーム保存
        times = [0] * 2  # [ラスタライゼーション時間, 逆伝播時間]
        # カメラの内部パラメータ行列（K行列）
        # 焦点距離と主点（画像中心）を定義
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],  # fx, skew, cx
                [0, self.focal, self.H / 2],  # 0, fy, cy
                [0, 0, 1],                    # 0, 0, 1
            ],
            device=self.device,
        )

        # レンダリング手法を選択
        if model_type == "3dgs":
            rasterize_fnc = rasterization      # 3Dガウシアンスプラッティング
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs  # 2Dガウシアンスプラッティング

        for iter in range(iterations):
            start = time.time()

            # 3Dガウシアンを2D画像にラスタライゼーション
            renders = rasterize_fnc(
                self.means,  # 3D位置
                self.quats / self.quats.norm(dim=-1, keepdim=True),  # 正規化された回転クォータニオン
                self.scales,  # スケール
                torch.sigmoid(self.opacities),  # シグモイドで[0,1]に変換された不透明度
                torch.sigmoid(self.rgbs),       # シグモイドで[0,1]に変換されたRGB色
                self.viewmat[None],  # ビュー行列（バッチ次元追加）
                K[None],            # 内部パラメータ行列（バッチ次元追加）
                self.W,             # 画像幅
                self.H,             # 画像高さ
                packed=False,       # テンソル形式の指定
            )[0]
            out_img = renders[0]  # レンダリング結果の画像
            torch.cuda.synchronize()  # GPU処理の同期
            times[0] += time.time() - start  # ラスタライゼーション時間を記録
            # レンダリング画像と目標画像の平均二乗誤差を計算
            loss = mse_loss(out_img, self.gt_image)
            # 勾配を初期化
            optimizer.zero_grad()
            start = time.time()
            # 逆伝播で勾配を計算
            loss.backward()
            torch.cuda.synchronize()  # GPU処理の同期
            times[1] += time.time() - start  # 逆伝播時間を記録
            # パラメータを更新
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            # 5イテレーション毎に画像を保存（アニメーション用）
            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # 保存されたフレームをGIFアニメーションとして出力
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,  # フレーム間隔（1/20秒）
                loop=0,      # 無限ループ
            )
        # パフォーマンス統計を出力
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    """画像ファイルをPyTorchテンソルに変換
    
    Args:
        image_path: 画像ファイルのパス
    
    Returns:
        [H, W, 3]形状のRGBテンソル（値域0-1）
    """
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()  # [0,1]範囲に正規化
    # [C, H, W] -> [H, W, C]に変換し、RGB成分のみ取得
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    """メイン関数：ガウシアンスプラッティングによる画像フィッティング
    
    Args:
        height: 画像の高さ
        width: 画像の幅
        num_points: 使用するガウシアンの数
        save_imgs: 訓練過程をGIFで保存するか
        img_path: 目標画像のパス（Noneの場合はテスト画像を生成）
        iterations: 最適化のイテレーション数
        lr: 学習率
        model_type: 使用するモデル（3dgsまたは2dgs）
    """
    if img_path:
        # 指定された画像を読み込み
        gt_image = image_path_to_tensor(img_path)
    else:
        # テスト用の簡単な画像を生成（左上が赤、右下が青）
        gt_image = torch.ones((height, width, 3)) * 1.0
        # 左上を赤色に設定
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        # 右下を青色に設定
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    # トレーナーを初期化して訓練を実行
    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main)
