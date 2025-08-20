# SACRED アーキテクチャの参考論文・手法

## 実装から推測される参考論文・手法

### 1. ChemBERTa の使用
```python
class ChemBERTaEncoder(nn.Module):
    model_name: str = "DeepChem/ChemBERTa-77M-MTR"
```
**参考論文**: 
- **ChemBERTa** (Chithrananda et al., 2020)
- "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"
- SMILES文字列の事前学習済み言語モデル

### 2. Conditional VAE (CVAE) アーキテクチャ
```python
class ConditionalVAE(nn.Module):
    def encode(self, x, c): # 条件付きエンコーディング
    def reparameterize(self, mu, logvar): # 再パラメータ化トリック
```
**参考論文**:
- **CVAE** (Sohn et al., 2015) 
- "Learning Structured Output Representation using Deep Conditional Generative Models"
- **Junction Tree VAE** (Jin et al., 2018)
- "Junction Tree Variational Autoencoder for Molecular Graph Generation"

### 3. Graph Neural Network (GNN) for Scaffold
```python
class ScaffoldGNN(nn.Module):
    self.conv1 = GCNConv(node_features, hidden_dim)
```
**参考論文**:
- **GCN** (Kipf & Welling, 2017)
- "Semi-Supervised Classification with Graph Convolutional Networks"
- **MolGAN** (De Cao & Kipf, 2018) - 分子生成でのGNN使用

### 4. Multi-Modal Fusion with Gating
```python
class MultiModalFusion(nn.Module):
    self.gate = nn.Sequential(nn.Linear(...), nn.Sigmoid())
    output = gate * combined + (1 - gate) * structural
```
**参考論文**:
- **Gated Multimodal Units** (Arevalo et al., 2017)
- "Gated Multimodal Units for Information Fusion"
- **MMVAE** (Shi et al., 2019) - マルチモーダルVAE

### 5. Transformer Decoder for SMILES
```python
class SMILESDecoder(nn.Module):
    self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
```
**参考論文**:
- **Transformer** (Vaswani et al., 2017) "Attention is All You Need"
- **MolGPT** (Bagal et al., 2021) - Transformer for molecular generation

### 6. Scaffold-Constrained Generation
**類似手法**:
- **SCAFF** (Lim et al., 2020)
- "Scaffold-based molecular design with a graph generative model"
- **SCGVAE** (Liu et al., 2021)
- "Scaffold-Constrained Molecular Generation"

### 7. ADMET-aware Generation  
**類似手法**:
- **MolDQN** (Zhou et al., 2019)
- "Optimization of Molecules via Deep Reinforcement Learning"
- **MIMOSA** (Skalic et al., 2019)
- "Shape-Based Generative Modeling for de Novo Drug Design"

## 統合アプローチの独自性

SACREDは以下の要素を組み合わせた独自のアーキテクチャ：

1. **3つのエンコーダの統合**
   - ChemBERTa (SMILES)
   - GNN (Scaffold)  
   - Attention (ADMET)

2. **階層的融合メカニズム**
   - 構造融合 → 特性統合
   - ゲート機構による動的重み付け

3. **条件付きVAE + Transformer**
   - 潜在空間での制約表現
   - 自己回帰的SMILES生成

## 最も近い既存研究

### 1. **C5T5** (Rothchild et al., 2021)
- "C5T5: Controllable Generation of Organic Molecules with Transformers"
- 制約付き分子生成でTransformer使用

### 2. **FREED** (Yang et al., 2021)  
- "Fragment-based Ligand Generation Guided by Geometric Deep Learning on Protein-Ligand Structure"
- マルチモーダル融合の類似アプローチ

### 3. **MoLeR** (Maziarz et al., 2022)
- "Learning to Extend Molecular Scaffolds with Structural Motifs"
- Scaffold拡張の類似手法

### 4. **DiffSBDD** (Schneuing et al., 2022)
- "Structure-based Drug Design with Equivariant Diffusion Models"
- 制約付き生成（ただし拡散モデル）

## SACREDの位置づけ

**独自の貢献**:
1. ChemBERTa + GNN + Attention の3モーダル統合
2. 階層的ゲート融合メカニズム
3. Scaffold制約とADMET制約の同時最適化
4. VAEとTransformerの組み合わせ

**最も類似**: Junction Tree VAE + C5T5 のハイブリッド的アプローチ

このアーキテクチャは2020-2022年頃の複数の最先端手法を統合・改良したものと推測されます。