"""
SACRED: Scaffold-Constrained ADMET-aware Conditional Encoder-Decoder
International Conference Ready Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import logging

logger = logging.getLogger(__name__)


class ChemBERTaEncoder(nn.Module):
    """Pre-trained ChemBERTa encoder for molecular SMILES"""
    
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR", freeze: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.model.config.hidden_size
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :]


class ScaffoldGNN(nn.Module):
    """Graph Neural Network for scaffold encoding"""
    
    def __init__(self, node_features: int = 74, hidden_dim: int = 256, output_dim: int = 384):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Graph convolutions with skip connections
        h1 = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.batch_norm2(self.conv2(h1, edge_index)))
        h2 = self.dropout(h2)
        
        h3 = self.conv3(h2, edge_index)
        
        # Global pooling
        out = global_mean_pool(h3, batch) + global_max_pool(h3, batch)
        return out


class PropertyEncoder(nn.Module):
    """ADMET property encoder with attention mechanism"""
    
    def __init__(self, num_properties: int = 13, hidden_dim: int = 256, output_dim: int = 384):
        super().__init__()
        
        # Individual property embeddings
        self.property_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_properties)
        ])
        
        # Attention mechanism for property importance
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * num_properties, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, properties: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = properties.shape[0]
        
        # Encode each property
        property_encodings = []
        for i, encoder in enumerate(self.property_embeddings):
            prop_encoding = encoder(properties[:, i:i+1])
            property_encodings.append(prop_encoding)
        
        # Stack encodings
        prop_tensor = torch.stack(property_encodings, dim=1)  # [B, num_props, hidden_dim]
        
        # Apply attention
        attended, _ = self.attention(prop_tensor, prop_tensor, prop_tensor, key_padding_mask=mask)
        
        # Flatten and project
        flattened = attended.reshape(batch_size, -1)
        output = self.output_projection(flattened)
        
        return output


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion of multiple modalities"""
    
    def __init__(self, mol_dim: int = 384, scaffold_dim: int = 384, property_dim: int = 384, 
                 latent_dim: int = 512):
        super().__init__()
        
        # First level: Molecule-Scaffold fusion
        self.mol_scaffold_fusion = nn.Sequential(
            nn.Linear(mol_dim + scaffold_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Second level: Add property constraints
        self.property_fusion = nn.Sequential(
            nn.Linear(latent_dim + property_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        
    def forward(self, mol_emb: torch.Tensor, scaffold_emb: torch.Tensor, 
                property_emb: torch.Tensor) -> torch.Tensor:
        
        # Level 1: Structural fusion
        structural = self.mol_scaffold_fusion(torch.cat([mol_emb, scaffold_emb], dim=-1))
        
        # Level 2: Property integration
        combined = self.property_fusion(torch.cat([structural, property_emb], dim=-1))
        
        # Gated output
        gate = self.gate(combined)
        output = gate * combined + (1 - gate) * structural
        
        return output


class ConditionalVAE(nn.Module):
    """Conditional VAE core with disentangled latent space"""
    
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, 
                 condition_dim: int = 512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # Latent parameters
        self.mu_layer = nn.Linear(512, latent_dim)
        self.logvar_layer = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


class SMILESDecoder(nn.Module):
    """Transformer-based SMILES decoder"""
    
    def __init__(self, latent_dim: int = 768, vocab_size: int = 100, max_length: int = 512):
        super().__init__()
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, latent_dim)
        self.position_embedding = nn.Embedding(max_length, latent_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Output projection
        self.output_projection = nn.Linear(latent_dim, vocab_size)
        
    def forward(self, latent: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = latent.shape[0]
        
        if target is not None:
            # Training mode
            seq_len = target.shape[1]
            
            # Token and position embeddings
            token_emb = self.token_embedding(target)
            pos_ids = torch.arange(seq_len, device=latent.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)
            
            # Combine embeddings
            decoder_input = token_emb + pos_emb
            
            # Create causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(latent.device)
            
            # Decode
            latent_expanded = latent.unsqueeze(1).expand(-1, seq_len, -1)
            output = self.transformer(decoder_input, latent_expanded, tgt_mask=mask)
            
        else:
            # Generation mode (autoregressive)
            output = self._generate(latent)
        
        return self.output_projection(output) if target is not None else output
    
    def _generate(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        device = latent.device
        
        # Start with BOS token (token ID 0)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(self.max_length - 1):
            # Get embeddings
            token_emb = self.token_embedding(generated)
            pos_ids = torch.arange(generated.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)
            
            decoder_input = token_emb + pos_emb
            
            # Decode
            latent_expanded = latent.unsqueeze(1).expand(-1, generated.shape[1], -1)
            output = self.transformer(decoder_input, latent_expanded)
            
            # Get next token
            next_token_logits = self.output_projection(output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class SACRED(nn.Module):
    """Main SACRED model"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        config = config or {}
        
        # Dimensions
        self.mol_dim = config.get('mol_dim', 384)
        self.scaffold_dim = config.get('scaffold_dim', 384)
        self.property_dim = config.get('property_dim', 384)
        self.fusion_dim = config.get('fusion_dim', 512)
        self.latent_dim = config.get('latent_dim', 256)
        self.vocab_size = config.get('vocab_size', 100)
        
        # Encoders
        self.mol_encoder = ChemBERTaEncoder(freeze=config.get('freeze_chemberta', True))
        self.scaffold_encoder = ScaffoldGNN(output_dim=self.scaffold_dim)
        self.property_encoder = PropertyEncoder(output_dim=self.property_dim)
        
        # Fusion
        self.fusion = HierarchicalFusion(
            mol_dim=self.mol_dim,
            scaffold_dim=self.scaffold_dim,
            property_dim=self.property_dim,
            latent_dim=self.fusion_dim
        )
        
        # CVAE
        self.cvae = ConditionalVAE(
            input_dim=self.fusion_dim,
            latent_dim=self.latent_dim,
            condition_dim=self.fusion_dim
        )
        
        # Decoder
        self.decoder = SMILESDecoder(
            latent_dim=self.latent_dim + self.fusion_dim,
            vocab_size=self.vocab_size
        )
        
        # Projection layers
        self.mol_projection = nn.Linear(self.mol_encoder.hidden_size, self.mol_dim)
        
    def forward(self, 
                smiles: List[str],
                scaffold_graphs: Batch,
                properties: torch.Tensor,
                target_smiles: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode inputs
        mol_emb = self.mol_encoder(smiles)
        mol_emb = self.mol_projection(mol_emb)
        
        scaffold_emb = self.scaffold_encoder(
            scaffold_graphs.x,
            scaffold_graphs.edge_index,
            scaffold_graphs.batch
        )
        
        property_emb = self.property_encoder(properties)
        
        # Hierarchical fusion
        fused = self.fusion(mol_emb, scaffold_emb, property_emb)
        
        # CVAE
        recon, mu, logvar = self.cvae(fused, fused)
        
        # Decode to SMILES
        z = self.cvae.reparameterize(mu, logvar)
        decoder_input = torch.cat([z, fused], dim=-1)
        output = self.decoder(decoder_input, target_smiles)
        
        return {
            'output': output,
            'reconstruction': recon,
            'mu': mu,
            'logvar': logvar,
            'fused_representation': fused
        }
    
    def generate(self,
                 scaffold_graphs: Batch,
                 properties: torch.Tensor,
                 num_samples: int = 10,
                 temperature: float = 1.0) -> List[torch.Tensor]:
        
        self.eval()
        with torch.no_grad():
            # Encode scaffold and properties
            scaffold_emb = self.scaffold_encoder(
                scaffold_graphs.x,
                scaffold_graphs.edge_index,
                scaffold_graphs.batch
            )
            property_emb = self.property_encoder(properties)
            
            # Create dummy molecular embedding (zeros for generation)
            batch_size = properties.shape[0]
            mol_emb = torch.zeros(batch_size, self.mol_dim, device=properties.device)
            
            # Fuse
            fused = self.fusion(mol_emb, scaffold_emb, property_emb)
            
            # Sample from latent space
            generated = []
            for _ in range(num_samples):
                z = torch.randn(batch_size, self.latent_dim, device=properties.device) * temperature
                decoder_input = torch.cat([z, fused], dim=-1)
                output = self.decoder(decoder_input)
                generated.append(output)
            
        return generated


class SACREDLoss(nn.Module):
    """Multi-objective loss function for SACRED"""
    
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.beta = beta  # KL weight
        self.gamma = gamma  # Reconstruction weight
        self.delta = delta  # Property prediction weight
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstruction'], targets['fused_input'])
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        
        # SMILES generation loss
        if 'output' in outputs and 'target_smiles' in targets:
            gen_loss = F.cross_entropy(
                outputs['output'].reshape(-1, outputs['output'].shape[-1]),
                targets['target_smiles'].reshape(-1)
            )
        else:
            gen_loss = torch.tensor(0.0, device=outputs['mu'].device)
        
        # Total loss
        total_loss = gen_loss + self.beta * kl_loss + self.gamma * recon_loss
        
        return {
            'total': total_loss,
            'generation': gen_loss,
            'kl': kl_loss,
            'reconstruction': recon_loss
        }