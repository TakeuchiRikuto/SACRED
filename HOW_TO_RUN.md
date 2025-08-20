# SACRED モデルの実行方法

## 1. 環境構築

### 必要な環境

### インストール
## 2. データ準備

### 方法1: サンプルデータ生成（テスト用）

```bash
# 1000個のサンプル分子データを生成
python prepare_data.py --mode sample --output data --num_samples 1000
```

これにより以下のファイルが生成されます：
- `data/train.jsonl` (800サンプル)
- `data/val.jsonl` (100サンプル) 
- `data/test.jsonl` (100サンプル)

### 方法2: ChEMBLデータ使用（本格的な学習用）

```bash
# ChEMBL CSVファイルから準備
python prepare_data.py --mode chembl --input chembl_data.csv --output data --max_samples 10000
```

## 3. モデル学習

### 基本的な学習

```bash
python train.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir checkpoints
```

### GPUを使用した学習

```bash
python train.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --epochs 100 \
    --batch_size 64 \
    --lr 5e-5 \
    --device cuda \
    --output_dir checkpoints
```

学習中の出力：
```
Using device: cuda
Model parameters: 124,567,890
Loaded 800 molecules from data/train.jsonl
Loaded 100 molecules from data/val.jsonl

=== Epoch 1/50 ===
Training: 100%|████████| 25/25 [00:45<00:00, 1.80s/it]
Training loss: 3.4567

=== Epoch 5/50 ===
Evaluating: 100%|████████| 4/4 [00:05<00:00, 1.25s/it]
Validation metrics: {'validity': 0.723, 'uniqueness': 0.891, ...}
Saved best model with score: 0.7456
```

## 4. 分子生成

### 基本的な生成

```bash
# ベンゼン環をscaffoldとして10分子生成
python generate.py \
    --model checkpoints/best_model.pt \
    --scaffold "c1ccccc1" \
    --num_molecules 10 \
    --output generated.csv
```

### プロパティ制約付き生成

```bash
# 分子量300-400、LogP 2-3の制約で生成
python generate.py \
    --model checkpoints/best_model.pt \
    --scaffold "c1ccccc1" \
    --properties '{"MW": [300, 400], "LogP": [2, 3], "HBA": [2, 5]}' \
    --num_molecules 50 \
    --temperature 0.8 \
    --output constrained_molecules.csv \
    --visualize
```

出力例：
```
=== Generation Summary ===
Valid molecules: 47/50 (94.0%)
Scaffold retained: 44/47 (93.6%)
MW satisfied: 41/47 (87.2%)
LogP satisfied: 39/47 (83.0%)
HBA satisfied: 45/47 (95.7%)
```

### 生成されたファイル
- `constrained_molecules.csv`: 生成分子のSMILESとプロパティ
- `constrained_molecules_grid.png`: 分子構造の可視化画像

## 5. 実験評価

### アブレーションスタディ実行

```bash
cd experiments
python ablation_study.py
```

結果：
- `results/ablation_study.csv`: 各コンポーネントの貢献度
- `results/ablation_study.png`: 可視化グラフ

## 6. トラブルシューティング

### よくあるエラーと対処法

#### 1. CUDA out of memory
```bash
# バッチサイズを小さくする
python train.py --batch_size 16 ...
```

#### 2. RDKit import error
```bash
# conda環境でRDKitをインストール
conda install -c conda-forge rdkit
```

#### 3. PyTorch Geometric エラー
```bash
# PyTorch バージョンに合わせてインストール
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 7. 簡単なテスト実行

すぐに動作確認したい場合：

```bash
# 1. サンプルデータ生成（1分）
python prepare_data.py --mode sample --num_samples 100

# 2. 短時間学習（5分）
python train.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --epochs 5 \
    --batch_size 8

# 3. 分子生成（1分）
python generate.py \
    --model checkpoints/best_model.pt \
    --num_molecules 5 \
    --output test_molecules.csv
```

## 8. Jupyter Notebookでの使用

```python
# Notebook内での使用例
import sys
sys.path.append('/Users/rikutotakeuchi/my_labo/my_research/SACRED')

from generate import MoleculeGenerator

# モデル初期化
generator = MoleculeGenerator('checkpoints/best_model.pt')

# 生成
molecules = generator.generate(
    scaffold='c1ccccc1',
    properties={'MW': [300, 400], 'LogP': [2, 3]},
    num_molecules=10
)

# 評価
results = generator.evaluate_generated(molecules, 'c1ccccc1')
for r in results:
    if r['valid']:
        print(f"SMILES: {r['smiles']}, MW: {r['MW']:.1f}, LogP: {r['LogP']:.2f}")
```

## 9. 高度な使用法

### カスタム設定ファイル

`config.yaml`を作成：
```yaml
mol_dim: 384
scaffold_dim: 384
property_dim: 384
fusion_dim: 512
latent_dim: 256
vocab_size: 100
freeze_chemberta: true
```

```bash
python train.py --config config.yaml ...
```

### バッチ処理

複数の制約セットで生成：
```python
constraints = [
    {'scaffold': 'c1ccccc1', 'properties': {'MW': [300, 400]}},
    {'scaffold': 'c1ncccn1', 'properties': {'LogP': [1, 3]}},
]

for c in constraints:
    molecules = generator.generate(**c)
    # 処理...
```

## サポート

問題が発生した場合は、エラーメッセージ全体を共有してください。