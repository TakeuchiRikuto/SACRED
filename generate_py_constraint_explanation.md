# generate.py の制約付き分子生成の実装方針

## 1. 制約の種類と設定方法

### 1.1 Scaffold（骨格構造）制約
```python
# コマンドライン引数で指定
--scaffold "c1ccccc1"  # ベンゼン環を骨格として指定
```

**実装の流れ（49-69行目）:**
```python
def generate(self, scaffold: str = None, ...):
    # 1. Scaffold SMILES → グラフ構造変換
    scaffold_graph_dict = self.featurizer.smiles_to_graph(scaffold)
    
    # 2. torch_geometric.Data形式に変換
    scaffold_graph = Data(x=scaffold_graph_dict['x'], 
                         edge_index=scaffold_graph_dict['edge_index'])
    
    # 3. バッチ化してGPUへ
    scaffold_batch = Batch.from_data_list([scaffold_graph]).to(self.device)
```

**制約の働き方:**
- ScaffoldGNNがグラフ構造を384次元ベクトルにエンコード
- このベクトルが生成の「条件」としてモデル全体に伝播
- Decoderは、この骨格情報を保持しながら新分子を生成

### 1.2 ADMET特性制約
```python
# コマンドライン引数でJSON形式で指定
--properties '{"MW": [200, 400], "LogP": [1, 3]}'
```

**実装の流れ（69-86行目）:**
```python
if properties is None:
    # デフォルト: 全特性を中間値(0.5)に設定
    property_values = [0.5] * 13
else:
    # 指定された範囲の中点を計算
    for prop_name in PROPERTY_FUNCTIONS.keys():
        if prop_name in properties:
            min_val, max_val = properties[prop_name]
            value = (min_val + max_val) / 2  # 範囲の中点
            
            # 正規化（0-1の範囲に変換）
            range_min, range_max = PROPERTY_RANGES[prop_name]
            normalized = (value - range_min) / (range_max - range_min)
```

**13個のADMET特性:**
1. MW: 分子量 (0-500)
2. LogP: 脂溶性 (-5 to 5)
3. HBA: 水素結合受容体数 (0-10)
4. HBD: 水素結合供与体数 (0-5)
5. TPSA: 極性表面積 (0-140)
6. RotatableBonds: 回転可能結合数 (0-10)
7. AromaticRings: 芳香環数 (0-4)
8. Heteroatoms: ヘテロ原子数 (0-10)
9. RingCount: 環構造数 (0-6)
10. FractionCsp3: sp3炭素の割合 (0-1)
11. SaturatedRings: 飽和環数 (0-4)
12. AliphaticRings: 脂肪族環数 (0-4)
13. QED: 薬物様性スコア (0-1)

## 2. 制約情報の統合メカニズム

### 2.1 モデル内での制約処理
```python
# generate関数内（89-95行目）
with torch.no_grad():
    generated_tokens = self.model.generate(
        scaffold_graphs=scaffold_batch,  # 骨格制約
        properties=property_tensor,      # ADMET制約
        num_samples=num_molecules,
        temperature=temperature          # 多様性制御
    )
```

### 2.2 SACRED モデルの制約適用フロー
```
1. 入力エンコーディング
   ├── Scaffold → ScaffoldGNN → 384次元
   └── Properties → PropertyEncoder → 384次元
           ↓
2. MultiModalFusion
   - 2つの制約を段階的に融合
   - ゲート機構で重要度を動的調整
           ↓
3. Conditional VAE
   - 制約を条件として潜在空間を学習
   - z = μ + ε * σ （制約条件付き）
           ↓
4. SMILES Decoder
   - 制約を満たすSMILES文字列を生成
   - Transformerが制約情報を考慮しながら自己回帰的に生成
```

## 3. 生成後の制約充足度評価

### 3.1 evaluate_generated関数（106-151行目）
```python
def evaluate_generated(self, smiles_list, scaffold, properties):
    for smiles in smiles_list:
        # 1. 分子の妥当性チェック
        mol = Chem.MolFromSmiles(smiles)
        
        # 2. Scaffold保持率の計算
        gen_scaffold = ScaffoldExtractor.extract_scaffold(smiles)
        similarity = calculate_scaffold_similarity(gen_scaffold, scaffold)
        result['scaffold_retained'] = similarity > 0.7  # 70%以上で保持と判定
        
        # 3. ADMET特性の制約充足チェック
        for prop_name in properties:
            min_val, max_val = properties[prop_name]
            actual_value = calculated_properties[prop_name]
            result[f'{prop_name}_satisfied'] = min_val <= actual_value <= max_val
```

## 4. 温度パラメータによる多様性制御

```python
--temperature 0.8  # デフォルト値
```

- **低温度 (0.1-0.5)**: 決定的、保守的な生成
- **中温度 (0.6-0.9)**: バランスの取れた生成
- **高温度 (1.0-2.0)**: より多様だが制約を外れやすい

## 5. 実行例と期待される動作

### 基本的な生成
```bash
python generate.py --model checkpoints/best_model.pt --num_molecules 100
```
→ 制約なしで100分子生成

### 骨格制約付き生成
```bash
python generate.py \
  --model checkpoints/best_model.pt \
  --scaffold "c1ccc2c(c1)OCO2" \  # ベンゾジオキソール骨格
  --num_molecules 50
```
→ 指定骨格を保持した50分子生成

### ADMET制約付き生成
```bash
python generate.py \
  --model checkpoints/best_model.pt \
  --properties '{"MW": [250, 350], "LogP": [2, 4], "HBA": [3, 7]}' \
  --num_molecules 100
```
→ 分子量250-350、LogP 2-4、水素結合受容体3-7個の条件で生成

### 複合制約
```bash
python generate.py \
  --model checkpoints/best_model.pt \
  --scaffold "c1ccccc1" \
  --properties '{"MW": [200, 400], "LogP": [1, 3]}' \
  --temperature 0.7 \
  --num_molecules 50 \
  --visualize
```
→ 骨格とADMET両方の制約を満たす分子を生成し、構造を可視化

## 6. なぜ今回Scaffold保持率が0%だったか

### 理由：
1. **サンプルデータでの訓練**: toy_dataは小規模で、骨格保持の学習が不十分
2. **モデルの未収束**: 限られたエポック数で複雑な制約学習が困難
3. **簡略化された実装**: SimpleMolecularFeaturizer等が完全な特徴抽出をしていない

### 本番環境で期待される改善：
- 大規模データセット（ChEMBL等）での訓練
- 長時間の学習（200+ epochs）
- 完全なRDKit/DeepChem統合
- ファインチューニングによる特定骨格への特化

## 7. 出力ファイル (generated_molecules.csv)

CSVには以下の情報が含まれる：
```csv
smiles,valid,canonical_smiles,scaffold_similarity,scaffold_retained,MW,LogP,HBA,HBD,...,MW_satisfied,LogP_satisfied,...
```

- **smiles**: 生成されたSMILES
- **valid**: 化学的妥当性
- **canonical_smiles**: 正規化SMILES
- **scaffold_similarity**: 骨格類似度（0-1）
- **scaffold_retained**: 骨格保持フラグ
- **各ADMET値**: 実際の計算値
- **X_satisfied**: 各制約の充足フラグ