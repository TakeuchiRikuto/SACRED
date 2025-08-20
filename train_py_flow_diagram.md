# train.py 実装フロー図

## メイン処理フロー

```
main() 開始
    │
    ├── 引数パース (argparse)
    │   ├── --train_data: 訓練データパス
    │   ├── --val_data: 検証データパス
    │   ├── --epochs: エポック数 (デフォルト: 50)
    │   ├── --batch_size: バッチサイズ (デフォルト: 32)
    │   ├── --lr: 学習率 (デフォルト: 1e-4)
    │   └── --output_dir: 出力ディレクトリ (デフォルト: checkpoints)
    │
    ├── 設定ファイル読み込み (config.yaml)
    │
    ├── デバイス設定 (CUDA/CPU)
    │
    ├── モデル初期化
    │   └── SACRED(config) → GPUへ転送
    │
    ├── データ処理コンポーネント作成
    │   ├── MolecularFeaturizer()
    │   ├── ADMETCalculator()
    │   ├── SMILESTokenizer()
    │   └── DataCollator()
    │
    ├── データセット作成
    │   ├── MolecularDataset(train_data) 
    │   │   └── JSONLファイル読み込み → SMILES抽出
    │   └── MolecularDataset(val_data)
    │
    ├── DataLoader作成
    │   ├── 訓練用 (shuffle=True, num_workers=4)
    │   └── 検証用 (shuffle=False, num_workers=4)
    │
    ├── 最適化設定
    │   ├── Optimizer: AdamW(lr=1e-4)
    │   ├── Scheduler: CosineAnnealingLR
    │   ├── Criterion: SACREDLoss()
    │   └── Evaluator: ConstraintEvaluator()
    │
    └── 訓練ループ (epochs回繰り返し)
        │
        ├── train_epoch() 実行
        │   │
        │   └── バッチごとの処理
        │       ├── データをGPUへ転送
        │       ├── model.forward() 実行
        │       ├── 損失計算 (SACREDLoss)
        │       ├── バックプロパゲーション
        │       ├── 勾配クリッピング (max_norm=1.0)
        │       └── optimizer.step()
        │
        ├── 5エポックごと: evaluate() 実行
        │   ├── model.generate() で分子生成
        │   ├── メトリクス計算
        │   │   ├── validity (妥当性)
        │   │   ├── uniqueness (一意性)
        │   │   ├── diversity (多様性)
        │   │   └── scaffold_retention (骨格保持率)
        │   ├── スコア計算 (重み付け平均)
        │   └── ベストモデル保存 (best_model.pt)
        │
        ├── scheduler.step() (学習率調整)
        │
        └── 10エポックごと: チェックポイント保存
            └── checkpoint_epoch_N.pt
```

## データ処理フロー (MolecularDataset.__getitem__)

```
JSONLファイルの1行
    │
    ├── SMILES文字列抽出
    │
    ├── Scaffold抽出
    │   └── 失敗時: デフォルト 'c1ccccc1' (ベンゼン環)
    │
    ├── ADMET特性計算 (13次元)
    │   └── 失敗時: デフォルト [0.5, 0.5, ..., 0.5]
    │
    └── 辞書形式で返却
        ├── 'smiles': 元のSMILES
        ├── 'scaffold': 抽出した骨格
        ├── 'properties': ADMET特性ベクトル
        └── 'target_smiles': 教師データ用SMILES
```

## DataCollator によるバッチ変換

```
個別データのリスト
    │
    ├── SMILES文字列 → そのままリスト
    │
    ├── Scaffold → グラフ構造に変換
    │   └── Batch.from_data_list() でバッチ化
    │
    ├── Properties → Tensorに変換
    │   └── shape: [batch_size, 13]
    │
    └── Target SMILES → トークン化
        └── shape: [batch_size, seq_length]
```

## 出力ファイル生成タイミング

```
訓練中の出力:
    │
    ├── 5エポックごと
    │   └── best_model.pt (スコア改善時のみ)
    │
    ├── 10エポックごと  
    │   └── checkpoint_epoch_{N}.pt
    │
    └── 訓練完了時
        ├── final_model.pt
        └── training_history.png (学習曲線グラフ)
```