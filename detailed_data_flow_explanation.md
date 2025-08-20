# SACRED モデルのデータフロー詳細解説

## 1. 入力データの3つの要素

### 1.1 Scaffold グラフ構造
**形式**: Graph Neural Network用のグラフデータ
```python
# ScaffoldGNN クラス (sacred_model.py:51-77)
入力:
- x: ノード特徴量テンソル (次元: [num_nodes, 74])
  - 74次元: 原子の種類、電荷、結合情報などの化学的特徴
- edge_index: エッジ情報 (次元: [2, num_edges])
  - 原子間の結合を表す隣接行列
- batch: バッチ処理用のノード割り当て情報

処理フロー:
1. GCNConv層で3層のグラフ畳み込み
2. 各層でBatchNormとDropout適用
3. global_mean_pool + global_max_poolでグラフ全体を384次元ベクトルに集約
```

### 1.2 ADMET特性ベクトル
**形式**: 13次元の正規化された物性値
```python
# PropertyEncoder クラス (sacred_model.py:80-115)
入力:
- 13個のADMET特性（各1次元、0-1に正規化）:
  - MW (分子量)
  - LogP (脂溶性)
  - HBA (水素結合受容体数)
  - HBD (水素結合供与体数)
  - TPSA (極性表面積)
  - RotatableBonds (回転可能結合数)
  - など

処理フロー:
1. 各特性を個別にEmbedding (1→256次元)
2. Multi-head Attention (4ヘッド)で特性間の相互作用をモデル化
3. 最終的に384次元ベクトルに変換
```

### 1.3 ターゲットSMILES（教師データ）
**形式**: トークン化されたSMILES文字列
```python
# ChemBERTaEncoder クラス (sacred_model.py:20-48)
入力:
- SMILES文字列のリスト
  例: ["CC(C)Cc1ccc(cc1)C(C)C(=O)O"]

処理フロー:
1. ChemBERTa tokenizerでトークン化（最大512トークン）
2. 事前学習済みChemBERTaモデルで埋め込み
3. [CLS]トークンの384次元ベクトルを取得
```

## 2. 融合処理（MultiModalFusion）

```python
# MultiModalFusion クラス (sacred_model.py:118-172)
3つの入力を段階的に融合:

Level 1: 構造融合
- mol_emb (ChemBERTa) + scaffold_emb (GNN) を結合
- 768次元 → 512次元に変換

Level 2: 特性統合  
- structural + property_emb を結合
- 896次元 → 512次元に変換

ゲート機構:
- gate = σ(Linear(combined))
- output = gate * combined + (1-gate) * structural
- 重要度に応じて動的に融合比率を調整
```

## 3. 条件付きVAE処理

```python
# ConditionalVAE クラス (sacred_model.py:175-227)

エンコード:
入力: fused_representation (512次元) + condition (512次元)
↓
Encoder Network (2層MLP)
↓
μ (平均: 256次元), σ (分散: 256次元)

再パラメータ化:
z = μ + ε * σ  (ε ~ N(0,1))

デコード:
z (256次元) + condition (512次元)
↓
Decoder Network (3層MLP)
↓
再構成された表現 (512次元)
```

## 4. SMILES生成（SMILESDecoder）

```python
# SMILESDecoder クラス (sacred_model.py:230-310)

訓練時（Teacher Forcing）:
1. latent (768次元) + target_tokens を入力
2. Token Embedding + Position Embedding
3. Transformer Decoder (6層、8ヘッド)
4. Causal Maskで未来の情報を遮断
5. 各位置で次のトークンを予測（vocab_size=100）

生成時（Autoregressive）:
1. BOSトークンから開始
2. 各ステップで次のトークンを予測
3. 予測トークンを次の入力に追加
4. EOSトークンまたは最大長まで繰り返し
```

## 5. 損失計算の詳細

```python
# train.py:88-93 の損失計算
losses = criterion(outputs, targets)

SACREDLoss の内部:
1. 再構成損失: CrossEntropy(predicted_tokens, target_tokens)
2. KL損失: KL(N(μ,σ), N(0,1)) 
3. 制約損失: 
   - Scaffold類似度損失
   - ADMET特性の制約違反ペナルティ
4. 総合損失: 0.7*再構成 + 0.2*KL + 0.1*制約
```

## 6. データの流れ図

```
入力データ
    ├── SMILES文字列 ──→ ChemBERTa ──→ 分子埋め込み (384次元)
    ├── Scaffold構造 ──→ GNN ──────→ 骨格埋め込み (384次元)
    └── ADMET特性 ────→ Attention ──→ 特性埋め込み (384次元)
                            ↓
                    MultiModalFusion
                            ↓
                    融合表現 (512次元)
                            ↓
                    Conditional VAE
                            ↓
                    潜在表現 (256次元)
                            ↓
                    SMILES Decoder
                            ↓
                    生成SMILES文字列
```

## 7. バッチ処理の実装

```python
# train.py:74-86 のバッチ処理
for batch in dataloader:
    # GPU転送
    scaffold_graphs = batch['scaffold_graphs'].to(device)  # Batch化されたグラフ
    properties = batch['properties'].to(device)            # [batch_size, 13]
    target_tokens = batch['target_tokens'].to(device)      # [batch_size, seq_len]
    
    # フォワードパス
    outputs = model(
        smiles=batch['smiles'],        # 文字列リスト（CPU上）
        scaffold_graphs=scaffold_graphs,
        properties=properties,
        target_smiles=target_tokens
    )
```

このように、3つの異なるモダリティ（SMILES文字列、グラフ構造、数値特性）を統合して、制約を満たす新しい分子を生成するアーキテクチャになっています。