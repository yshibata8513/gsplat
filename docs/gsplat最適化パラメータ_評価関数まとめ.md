# GSplat 最適化パラメータと評価関数の詳細解析

本文書では、`simple_trainer.py`をbasic.shで実行した場合の最適化対象パラメータと評価関数について、曖昧さなく詳細に記載する。

## 最適化対象パラメータの分類

### 1. 3Dガウシアンの物理パラメータ（必須・メイン最適化対象）

#### 1.1 位置パラメータ（means）
- **パラメータ**: `self.splats["means"]` - [N, 3]
- **物理的意味**: 各ガウシアンの3D空間での位置座標
- **初期化方法**: 
  - SfMモード（デフォルト）: `parser.points` - **データセットにCOLMAPで事前計算されたSfM点群が含まれている**
    - 実際のパス: `data_dir/sparse/0/points3D.bin`（COLMAP形式）
    - SceneManagerにより`manager.points3D`として読み込み
    - **必須データ**: 3D位置座標 `points` [N, 3] + RGB色情報 `points_rgb` [N, 3]
    - **オプション**: 点群の再投影誤差情報 `points_err` [N]
    - 正規化処理後の実3D座標と初期RGB色を使用
  - ランダムモード: `init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)`
- **学習対象**: パラメータそのもの（絶対座標）
- **学習率**: `means_lr = 1.6e-4` × `scene_scale` × `sqrt(batch_size)`
- **スケジューラー**: 指数減衰（最終値は初期値の1%）
- **最適化器**: Adam（基本）/ SparseAdam（sparse_grad=True）/ SelectiveAdam（visible_adam=True）

#### 1.2 サイズパラメータ（scales）
- **パラメータ**: `self.splats["scales"]` - [N, 3] 
- **物理的意味**: 楕円体ガウシアンの各軸長（対数空間で格納）
- **初期化方法**: k-近傍法による自動決定
  ```python
  dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
  dist_avg = torch.sqrt(dist2_avg)
  scales = torch.log(dist_avg * init_scale)
  ```
- **学習対象**: 対数値（実際のサイズはtorch.exp(scales)で復元）
- **学習率**: `scales_lr = 5e-3` × `sqrt(batch_size)`
- **初期値係数**: `init_scale = 1.0`

#### 1.3 方向パラメータ（quats）
- **パラメータ**: `self.splats["quats"]` - [N, 4]
- **物理的意味**: クォータニオンによる3D回転（楕円体の方向）
- **初期化方法**: ランダム `torch.rand((N, 4))`
- **学習対象**: クォータニオン成分そのもの
- **学習率**: `quats_lr = 1e-3` × `sqrt(batch_size)`
- **正規化**: レンダリング時に内部的に正規化実行

#### 1.4 不透明度パラメータ（opacities）
- **パラメータ**: `self.splats["opacities"]` - [N,]
- **物理的意味**: 各ガウシアンの不透明度
- **初期化方法**: logit空間での初期化 `torch.logit(torch.full((N,), init_opacity))`
- **学習対象**: logit値（実際の不透明度はsigmoid(opacities)で復元）
- **学習率**: `opacities_lr = 5e-2` × `sqrt(batch_size)`
- **初期不透明度**: `init_opa = 0.1`

#### 1.5 色パラメータ（球面調和関数）

**基本色（sh0）**
- **パラメータ**: `self.splats["sh0"]` - [N, 1, 3]
- **物理的意味**: 球面調和関数0次項（環境光・基本色）
- **初期化方法**: RGB→SH変換 `colors[:, 0, :] = rgb_to_sh(rgbs)`
- **学習率**: `sh0_lr = 2.5e-3` × `sqrt(batch_size)`

**詳細色（shN）**
- **パラメータ**: `self.splats["shN"]` - [N, K-1, 3] （K = (sh_degree+1)²）
- **物理的意味**: 球面調和関数高次項（視点依存の詳細色）
- **初期化方法**: ゼロ初期化
- **学習率**: `shN_lr = sh0_lr / 20 = 1.25e-4` × `sqrt(batch_size)`
- **段階的活用**: `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)`

### 2. カメラ姿勢最適化パラメータ（オプション）

#### 2.1 カメラ姿勢調整
- **有効条件**: `pose_opt = True`
- **パラメータ**: `self.pose_adjust.embeds.weight` - [画像数, 9]
- **物理的意味**: **画像ごとの姿勢誤差修正量**
  - 位置誤差: [画像ID, 0:3] - 3D位置の修正ベクトル
  - 回転誤差: [画像ID, 3:9] - 6D回転表現での修正量
- **重要**: **各画像に対して独立した姿勢修正パラメータを保持**
  - 例：100枚の訓練画像 → [100, 9]の修正パラメータ行列
  - 各画像のimage_idに対応した修正量を学習
- **初期化方法**: ゼロ初期化 `torch.nn.init.zeros_(self.embeds.weight)`
- **学習対象**: **初期姿勢からの修正量**（絶対姿勢ではない）
- **学習率**: `pose_opt_lr = 1e-5` × `sqrt(batch_size)`
- **正則化**: L2正則化 `pose_opt_reg = 1e-6`
- **適用方法**: `camtoworlds = self.pose_adjust(camtoworlds, image_ids)`

#### 2.2 姿勢摂動（テスト用）
- **有効条件**: `pose_noise > 0.0`
- **パラメータ**: `self.pose_perturb.embeds.weight` - [画像数, 9]
- **物理的意味**: カメラキャリブレーション誤差のシミュレーション
- **初期化方法**: 正規分布 `torch.nn.init.normal_(std=pose_noise)`
- **学習対象**: なし（固定ノイズ）

### 3. 外観最適化パラメータ（実験的）

#### 3.1 画像固有外観エンベディング
- **有効条件**: `app_opt = True`
- **パラメータ**: `self.app_module.embeds.weight` - [画像数, app_embed_dim]
- **物理的意味**: **画像ごとの照明・色調変化**の学習表現
- **初期化方法**: デフォルト初期化（最終層はゼロ初期化）
- **学習対象**: エンベディングベクトルそのもの
- **学習率**: `app_opt_lr * sqrt(batch_size) * 10.0 = 1e-2`（エンベディング）
- **学習率**: `app_opt_lr * sqrt(batch_size) = 1e-3`（MLPヘッド）

#### 3.2 特徴量ベース色表現
- **パラメータ**: 
  - `self.splats["features"]` - [N, 32] ガウシアン特徴量
  - `self.splats["colors"]` - [N, 3] ベース色（logit空間）
- **物理的意味**: 複雑な材質特性の数値表現
- **学習率**: `sh0_lr`と同等

## 密度化戦略パラメータ（DefaultStrategy）

### ガウシアン数の動的変化プロセス

#### 初期ガウシアン数
- **データセット由来**: SfM点群の点数をそのまま使用（例：数万～数十万点）
- **実例**: `print("Model initialized. Number of GS:", len(self.splats["means"]))`で初期数を表示

#### 動的増減操作（学習中に実行）

**複製（duplicate）** - ガウシアン数増加
- **条件**: 高勾配 かつ 小3Dスケール
- **動作**: 既存ガウシアンを同位置にコピー
- **目的**: 点的な詳細表現の強化

**分割（split）** - ガウシアン数増加  
- **条件**: 高勾配 かつ 大3Dスケール
- **動作**: 1つのガウシアンを2つに分割
- **目的**: ぼやけた大きなガウシアンの詳細化

**削除（prune）** - ガウシアン数減少
- **条件**: 低不透明度 または 過大スケール
- **動作**: 不要ガウシアンの除去
- **目的**: メモリ効率化と過学習防止

### 動的ガウシアン管理の物理的制御
- **prune_opa**: 0.005 - 不透明度閾値（これ以下は削除）
- **grow_grad2d**: 0.0002 - 2D勾配閾値（詳細不足領域の判定）
- **grow_scale3d**: 0.01 - 小サイズ複製閾値（点的ガウシアンの増殖）
- **grow_scale2d**: 0.05 - 大サイズ分割閾値（ぼやけ解消）
- **prune_scale3d**: 0.1 - 大サイズ削除閾値（過大ガウシアン除去）
- **refine_start_iter**: 500 - 密度化開始時期
- **refine_stop_iter**: 15,000 - 密度化終了時期  
- **reset_every**: 3,000 - 不透明度リセット間隔
- **refine_every**: 100 - 密度化実行間隔

### 学習中のガウシアン数変化例
```
初期: 50,000点（SfM）
step 1000: 55,000点（複製・分割による増加）
step 5000: 120,000点（最大付近）
step 10000: 80,000点（削除により安定化）
```

## 評価関数の詳細

### 1. 主要損失関数

#### 1.1 L1損失
```python
l1loss = F.l1_loss(colors, pixels)
```
- **物理的意味**: ピクセル単位の絶対誤差
- **重み**: `(1.0 - ssim_lambda) = 0.8`

#### 1.2 SSIM損失
```python
ssimloss = 1.0 - fused_ssim(colors, pixels, padding="valid")
```
- **物理的意味**: 構造的類似性（人間の視覚に近い評価）
- **重み**: `ssim_lambda = 0.2`

#### 1.3 総合損失
```python
loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
```

### 2. 正則化項

#### 2.1 不透明度正則化（オプション）
```python
loss += opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
```
- **目的**: 不必要に不透明なガウシアンの抑制

#### 2.2 スケール正則化（オプション）
```python
loss += scale_reg * torch.exp(self.splats["scales"]).mean()
```
- **目的**: 過度に大きなガウシアンの抑制

#### 2.3 深度損失（実験的）
- **有効条件**: `depth_loss = True`
- **物理的意味**: 既知深度情報による3D構造制約
- **重み**: `depth_lambda = 1e-2`

### 3. 評価指標

#### 3.1 画質評価指標
- **PSNR**: Peak Signal-to-Noise Ratio（信号対雑音比）
- **SSIM**: Structural Similarity Index Measure（構造的類似性）
- **LPIPS**: Learned Perceptual Image Patch Similarity（知覚的類似性）
  - **ネットワーク**: AlexNet (`lpips_net = "alex"`)

#### 3.2 性能指標
- **レンダリング時間**: GPU同期による正確な測定
- **ガウシアン数**: モデルの複雑性指標
- **メモリ使用量**: CUDA最大メモリ割当量

## 学習フロー詳細

### 1. パラメータ更新順序
1. **前処理**: 密度化戦略による勾配情報収集
2. **損失計算**: L1 + SSIM + 正則化項
3. **バックプロパゲーション**: 勾配計算
4. **パラメータ更新**:
   - ガウシアンパラメータ（means, scales, quats, opacities, colors）
   - カメラ姿勢（オプション）
   - 外観エンベディング（オプション）
5. **後処理**: 密度化戦略による動的ガウシアン管理

### 2. 学習率スケジューリング
- **位置パラメータ**: 指数減衰（最終値1%）
- **その他パラメータ**: 固定学習率
- **バッチサイズ補正**: 全パラメータに`sqrt(batch_size)`を乗算

### 3. 最適化器選択
- **基本**: Adam（eps=1e-15/√BS, betas調整済み）
- **スパース勾配**: SparseAdam（メモリ効率化）
- **可視性最適化**: SelectiveAdam（可視ガウシアンのみ更新）

## 必須データセット構成

GSplatで学習を実行するために、データセットに含まれるべき**必須データ**を以下に詳述する。

### 1. ディレクトリ構造（COLMAP形式）
```
data_dir/
├── images/                    # 入力画像
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── sparse/0/                  # COLMAP復元結果
│   ├── cameras.bin            # カメラ内部パラメータ
│   ├── images.bin             # カメラ姿勢・画像メタデータ
│   └── points3D.bin           # SfM点群データ
└── poses_bounds.npy (オプション) # 前方向シーン用境界情報
```

### 2. 画像データ（images/）

#### 2.1 画像ファイル
- **形式**: JPG, PNG等の一般的画像形式
- **命名**: 任意（COLMAP内部で自動対応付け）
- **内容**: RGB画像（3チャンネル）
- **解像度**: 任意（factor設定でダウンサンプル可能）

### 3. カメラ内部パラメータ（cameras.bin）

#### 3.1 必須パラメータ（カメラごと）
- **カメラID**: 一意識別子
- **画像サイズ**: `width × height` [ピクセル]
- **焦点距離**: `fx, fy` [ピクセル]
- **主点**: `cx, cy` [ピクセル] 
- **カメラモデル**: 以下のいずれか
  - `SIMPLE_PINHOLE` (type=0): パラメータなし
  - `PINHOLE` (type=1): パラメータなし  
  - `SIMPLE_RADIAL` (type=2): `k1`
  - `RADIAL` (type=3): `k1, k2`
  - `OPENCV` (type=4): `k1, k2, p1, p2`
  - `OPENCV_FISHEYE` (type=5): `k1, k2, k3, k4`

#### 3.2 カメラ内部パラメータ行列構成
```python
K = [[fx,  0, cx],
     [ 0, fy, cy], 
     [ 0,  0,  1]]
```

### 4. カメラ姿勢・画像メタデータ（images.bin）

#### 4.1 必須データ（画像ごと）
- **画像ID**: 一意識別子
- **カメラID**: 対応するカメラの参照ID
- **画像ファイル名**: `images/`内の相対パス
- **カメラ姿勢**: 以下の座標系で定義

#### 4.2 カメラ姿勢の座標系定義
**COLMAP座標系**（右手座標系）:
- **回転行列**: `R` [3×3] - **World-to-Camera**変換
- **並進ベクトル**: `t` [3×1] - **World-to-Camera**変換
- **変換方向**: 世界座標点 → カメラ座標点

**GSplat内部使用形式**（逆変換）:
- **camtoworlds**: `np.linalg.inv(w2c_mats)` [4×4] - **Camera-to-World**変換
- **座標系**: OpenGL/OpenCV準拠
- **軸定義**: 
  - X軸: 右方向
  - Y軸: 下方向 
  - Z軸: 前方向（カメラ視線方向）

### 5. SfM点群データ（points3D.bin）

#### 5.1 必須データ（全体で1つの点群）
- **3D座標**: `points` [N, 3] - **世界座標系**での位置
- **RGB色**: `points_rgb` [N, 3] - 0-255の整数値
- **観測情報**: 各点を観測した画像ID・特徴点座標のリスト
- **再投影誤差**: `points_err` [N] （オプション）

#### 5.2 世界座標系定義
**COLMAP原座標系**:
- 任意の世界座標系（最初のカメラ基準等）
- 単位: メートル或いは任意単位

**正規化後座標系**（normalize=True時）:
- 原点: カメラ群の中心付近
- スケール: カメラ配置に基づく正規化
- 上方向: Z+軸（推定された上方向に整列）
- 座標変換: 主軸整列・反転補正を適用

### 6. オプションデータ

#### 6.1 境界情報（poses_bounds.npy）
- **形式**: NumPy配列 [画像数, 17]
- **内容**: カメラ姿勢(15要素) + 近・遠境界(2要素)
- **用途**: 前方向シーン（LLFF形式）の境界設定

#### 6.2 拡張メタデータ（ext_metadata.json）
```json
{
    "spiral_radius_scale": 1.0,
    "no_factor_suffix": false
}
```

### 7. データ整合性要件

#### 7.1 ID対応関係
- `cameras.bin`のカメラID ↔ `images.bin`の参照カメラID
- `images.bin`の画像ファイル名 ↔ `images/`内の実ファイル
- `points3D.bin`の観測リスト ↔ `images.bin`の画像ID

#### 7.2 座標系統一
- カメラ姿勢・点群座標が同一世界座標系で定義されていること
- COLMAP標準出力（右手座標系）に従うこと

#### 7.3 スケール統一
- カメラ姿勢の並進成分と点群座標が同一単位系であること
- 内部で自動正規化されるため、実際の単位は任意

### 8. データセット構築手順

#### 8.1 推奨ワークフロー
1. **COLMAP実行**: `colmap automatic_reconstructor`
2. **データ検証**: 画像・姿勢・点群の整合性確認  
3. **ディレクトリ配置**: 上記構造に従った配置
4. **GSplat実行**: 設定ファイルでdata_dirを指定


これらの仕様に従ってデータセットを構築することで、GSplatによる3Dガウシアンスプラッティング学習を曖昧さなく実行できる。

## 実行設定（basic.sh使用時）

### データセット固有設定
- **高解像度シーン**: `data_factor = 2`（bonsai, counter, kitchen, room）
- **標準解像度シーン**: `data_factor = 4`（garden, bicycle, stump）
- **テスト間隔**: `test_every = 8`（8枚ごとに1枚をテスト用）

### レンダリング設定
- **軌道タイプ**: `ellipse`（楕円軌道でのカメラ動作）
- **近クリップ平面**: `0.01`
- **遠クリップ平面**: `1e10`
- **カメラモデル**: `pinhole`（ピンホールカメラ）

この詳細な仕様により、gsplatの3Dガウシアンスプラッティング最適化プロセスの全貌を曖昧さなく理解できる。各パラメータは物理的意味と数学的制約の両面から設計されており、高品質な3D再構成とリアルタイムレンダリングを実現する。