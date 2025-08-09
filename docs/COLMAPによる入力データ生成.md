
## COLMAP automatic_reconstructor の詳細実行方法

**COLMAP automatic_reconstructor** は、入力画像から自動的に3D復元（SfM）を実行するワンコマンドツールです。

##### 8.2.1 必要な入力データ
- **画像のみ**: RGB画像ファイル（JPG, PNG等）
- **撮影要件**: 
  - 同一シーン・オブジェクトを複数視点から撮影
  - 隣接画像間で十分な重複領域（60-80%推奨）
  - 適度な画質（ブレ・ボケを最小限に）

##### 8.2.2 ディレクトリ準備
```bash
# プロジェクトフォルダ作成
mkdir my_dataset
cd my_dataset

# 画像フォルダ作成・配置
mkdir images
# 撮影画像をimages/に配置
cp /path/to/your/photos/*.jpg images/
```

##### 8.2.3 COLMAP実行コマンド
```bash
# 基本実行（GPU使用）
DATASET_PATH=/path/to/my_dataset
colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images

# GPU無しシステムでの実行
colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images \
    --use_gpu 0 \
    --SiftExtraction.use_gpu 0 \
    --SiftMatching.use_gpu 0

# 単一カメラでの撮影の場合
colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images \
    --single_camera 1
```

##### 8.2.4 生成される出力構造
実行成功後、以下の構造が自動生成されます：
```
my_dataset/
├── images/                    # 入力画像（元から存在）
│   ├── IMG_001.jpg
│   └── ...
├── database.db               # 特徴点データベース
├── sparse/0/                 # スパース復元結果（GSplat必須）
│   ├── cameras.bin           # カメラ内部パラメータ
│   ├── images.bin            # カメラ姿勢・画像メタデータ  
│   └── points3D.bin          # SfM点群データ
└── dense/0/ (オプション)      # 密復元結果
    ├── fused.ply
    ├── images/
    └── stereo/
```

##### 8.2.5 GSplat用データセット変換
COLMAP出力は既にGSplat対応形式のため、追加変換は不要：
```bash
# GSplat学習実行
python simple_trainer.py default \
    --data_dir /path/to/my_dataset \
    --result_dir results/my_scene
```

##### 8.2.6 トラブルシューティング
- **復元失敗**: 画像間の重複不足、特徴点不足
- **一部画像が復元されない**: カメラ姿勢推定失敗
- **点群が少ない**: テクスチャ不足、照明条件不良
- **スケール不正**: 単眼カメラのスケール曖昧性

これにより、**撮影画像のみから**GSplat学習に必要な完全なデータセット（カメラ姿勢＋3D点群）を自動生成できます。
