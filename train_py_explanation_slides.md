# train.py の詳細説明スライド

---

## スライド1: 概要
### train.py の役割
- **目的**: SACRED分子生成モデルの学習スクリプト
- **主要機能**: 
  - データローディング
  - モデル訓練
  - 検証・評価
  - チェックポイント保存

---

## スライド2: データ処理フロー
### MolecularDataset クラス (31-66行目)
```
入力データ (JSONL形式)
    ↓
SMILES分子構造の読み込み
    ↓
Scaffold（骨格構造）の抽出
    ↓
ADMET特性の計算（13個の物性値）
    ↓
バッチ処理用データ形式に変換
```

---

## スライド3: 学習プロセス
### train_epoch 関数 (69-103行目)

1. **前向き計算**
   - Scaffold グラフ構造
   - ADMET特性ベクトル
   - ターゲットSMILES（教師データ）

2. **損失計算**
   - SACREDLoss による複合損失
   - 再構成損失 + 制約損失

3. **逆伝播と最適化**
   - 勾配クリッピング (max nor　m: 1.0)
   - AdamW オプティマイザ

---

## スライド4: 評価メトリクス
### evaluate 関数 (106-143行目)

**生成分子の評価指標:**
- **Validity (妥当性)**: 化学的に正しい分子の割合
- **Uniqueness (一意性)**: 重複のない分子の割合  
- **Diversity (多様性)**: 生成分子の構造的多様性
- **Scaffold Retention**: 指定骨格の保持率

**スコア計算式:**
```
総合スコア = 0.3×妥当性 + 0.2×一意性 + 0.2×多様性 + 0.3×骨格保持率
```

---

## スライド5: 学習設定
### main 関数のパラメータ (146-156行目)

| パラメータ | デフォルト値 | 説明 |
|---------|----------|------|
| --epochs | 50 | 学習エポック数 |
| --batch_size | 32 | バッチサイズ |
| --lr | 1e-4 | 学習率 |
| --device | cuda | 計算デバイス |
| --output_dir | checkpoints | モデル保存先 |

---

## スライド6: 学習ループの詳細
### エポックごとの処理 (215-242行目)

```python
各エポックで:
1. 訓練データでモデル更新
2. 5エポックごとに検証評価
3. ベストモデルの保存（スコアが改善時）
4. 10エポックごとにチェックポイント保存
5. 学習率のコサインアニーリング
```

---

## スライド7: 出力ファイル
### 保存されるファイル

1. **best_model.pt**
   - 検証スコアが最高のモデル
   - 推論に使用推奨

2. **checkpoint_epoch_N.pt**
   - 10エポックごとの完全な状態
   - 学習再開用

3. **final_model.pt**
   - 最終エポックのモデル

4. **training_history.png**
   - 学習曲線のグラフ

---

## スライド8: 学習の実行方法
### コマンド例

```bash
python train.py \
  --train_data toy_data/train.jsonl \
  --val_data toy_data/val.jsonl \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir checkpoints
```

---

## スライド9: モデル生成の実行
### 学習済みモデルを使った分子生成

**使用ファイル**: `generate.py`

**実行例:**
```bash
# 基本的な生成
python generate.py \
  --model checkpoints/best_model.pt \
  --num_molecules 100 \
  --output generated_molecules.csv

# 制約付き生成
python generate.py \
  --model checkpoints/best_model.pt \
  --scaffold "c1ccccc1" \
  --properties '{"MW": [200, 400], "LogP": [1, 3]}' \
  --num_molecules 50 \
  --visualize
```

---

## スライド10: generate.py の主要機能

1. **MoleculeGenerator クラス**
   - 学習済みモデルのロード
   - 制約付き分子生成

2. **生成パラメータ**
   - scaffold: 保持したい骨格構造
   - properties: ADMET特性の制約範囲
   - temperature: 生成の多様性制御

3. **出力**
   - CSV形式の生成分子リスト
   - 評価メトリクス付き
   - 分子構造の可視化画像（オプション）