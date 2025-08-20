# generate.py の入力・出力仕様

## モデルへの入力（model.generate()に渡すもの）

### 1. 入力データ形式
```python
# generate.py の 90-95行目で実際に渡している入力
generated_tokens = self.model.generate(
    scaffold_graphs=scaffold_batch,  # torch_geometric.data.Batch
    properties=property_tensor,      # torch.Tensor
    num_samples=num_molecules,       # int
    temperature=temperature          # float
)
```

### 具体的な入力例

#### scaffold_graphs（必須）
```python
# Batchオブジェクト（グラフデータ）
scaffold_batch = {
    'x': torch.Tensor([batch_size * num_nodes, 74]),  # ノード特徴量
    'edge_index': torch.Tensor([2, batch_size * num_edges]),  # エッジ情報
    'batch': torch.Tensor([batch_size * num_nodes])  # バッチ割り当て
}

# 例: ベンゼン環 "c1ccccc1" の場合
# x: [6, 74] - 6個の炭素原子、各74次元の特徴
# edge_index: [2, 12] - 6個の結合×2方向
```

#### properties（必須）
```python
# 13次元の正規化されたADMET特性値
property_tensor = torch.Tensor([[0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.3, 0.8, 0.5, 0.4, 0.6, 0.5]])
# shape: [1, 13] （バッチサイズ1の場合）

# 各要素は0-1に正規化された値：
# [MW, LogP, HBA, HBD, TPSA, RotatableBonds, AromaticRings, 
#  Heteroatoms, RingCount, FractionCsp3, SaturatedRings, AliphaticRings, QED]
```

#### その他のパラメータ
```python
num_samples = 10      # 生成する分子数
temperature = 0.8     # サンプリング温度（0.1-2.0）
```

## モデルからの出力

### 1. model.generate()の戻り値
```python
# sacred_model.py の generate メソッドの出力
generated_tokens: List[torch.Tensor]
# 各テンソル: [batch_size, sequence_length]
# トークンID（0-99の整数）のシーケンス

# 例:
# [[0, 15, 23, 45, 67, 23, 45, 2, 1, ...]]  # 0=BOS, 1=EOS, 2-99=SMILES文字
```

### 2. トークンからSMILESへの変換
```python
# generate.py の 98-102行目
for tokens_batch in generated_tokens:
    tokens = tokens_batch[0]  # バッチから取り出し
    smiles = self.tokenizer.decode(tokens.cpu().numpy())
    # 例: "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
```

## 訓練時と生成時の違い

### 訓練時（train.py）
```python
# Teacher Forcingあり
outputs = model(
    smiles=batch['smiles'],          # 実際のSMILES文字列（教師データ）
    scaffold_graphs=scaffold_graphs,
    properties=properties,
    target_smiles=target_tokens      # 正解トークン列（教師データ）
)
# 出力: 損失計算用の内部表現
```

### 生成時（generate.py）
```python
# 自己回帰的生成（教師データなし）
generated = model.generate(
    scaffold_graphs=scaffold_graphs,  # 制約1: 骨格構造
    properties=properties,            # 制約2: ADMET特性
    num_samples=num_samples,
    temperature=temperature
)
# 出力: 生成されたトークン列
```

## 期待される生成結果

### 理想的な場合
```python
入力:
- scaffold: "c1ccccc1" (ベンゼン環)
- properties: {"MW": [200, 300], "LogP": [2, 3]}

期待される出力:
- "CC(C)c1ccccc1"      # イソプロピルベンゼン (MW: 120, LogP: 3.15)
- "CCCc1ccccc1"        # プロピルベンゼン (MW: 120, LogP: 3.18)
- "c1ccc(CC)cc1"       # エチルベンゼン (MW: 106, LogP: 3.15)
# → ベンゼン環を保持し、MW/LogP制約に近い分子
```

### 現在の結果（サンプルデータ）
```python
入力: 同上

実際の出力:
- ランダムなSMILES（制約を満たさない）
- Scaffold保持率: 0%
- 理由: 訓練データ不足、モデル未収束
```

## まとめ

**入力**: 
- グラフ構造（骨格）+ 13次元特性ベクトル（ADMET）

**処理**:
1. ScaffoldGNN → 384次元
2. PropertyEncoder → 384次元  
3. MultiModalFusion → 512次元
4. Conditional VAE → 256次元潜在表現
5. Transformer Decoder → トークン列

**出力**:
- SMILESトークン列 → デコード → SMILES文字列

訓練が十分であれば、入力の制約を満たすSMILESが生成されるはずです。