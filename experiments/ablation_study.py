"""
Ablation study and experimental validation for SACRED
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model.sacred_model import SACRED, SACREDLoss
from model.data_processing import (
    MolecularFeaturizer, ADMETCalculator, 
    ScaffoldExtractor, DataCollator, SMILESTokenizer
)
from evaluation.metrics import ConstraintEvaluator, PerformanceTracker

logger = logging.getLogger(__name__)


class AblationStudy:
    """Conduct ablation studies on SACRED components"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ConstraintEvaluator()
        
        # Define ablation configurations
        self.ablation_configs = {
            'full_model': {
                'description': 'Full SACRED model',
                'modifications': {}
            },
            'no_chemberta': {
                'description': 'Without ChemBERTa encoder',
                'modifications': {'use_chemberta': False}
            },
            'no_gnn': {
                'description': 'Without GNN scaffold encoder',
                'modifications': {'use_gnn': False}
            },
            'no_attention': {
                'description': 'Without attention in property encoder',
                'modifications': {'use_attention': False}
            },
            'no_hierarchical': {
                'description': 'Without hierarchical fusion',
                'modifications': {'use_hierarchical_fusion': False}
            },
            'no_cvae': {
                'description': 'Without CVAE (deterministic)',
                'modifications': {'use_cvae': False}
            },
            'simple_decoder': {
                'description': 'With simple MLP decoder',
                'modifications': {'decoder_type': 'mlp'}
            }
        }
    
    def create_ablated_model(self, config_name: str) -> SACRED:
        """Create model with ablated components"""
        config = self.base_config.copy()
        modifications = self.ablation_configs[config_name]['modifications']
        config.update(modifications)
        
        # Create modified model
        if config_name == 'no_chemberta':
            # Replace with random embeddings
            model = SACRED(config)
            model.mol_encoder = self._create_random_encoder()
        elif config_name == 'no_gnn':
            # Replace with MLP encoder
            model = SACRED(config)
            model.scaffold_encoder = self._create_mlp_encoder()
        elif config_name == 'no_attention':
            # Modify property encoder
            model = SACRED(config)
            model.property_encoder = self._create_simple_property_encoder()
        elif config_name == 'no_hierarchical':
            # Simple concatenation fusion
            model = SACRED(config)
            model.fusion = self._create_concat_fusion()
        elif config_name == 'no_cvae':
            # Deterministic model
            model = self._create_deterministic_model(config)
        elif config_name == 'simple_decoder':
            # MLP decoder
            model = SACRED(config)
            model.decoder = self._create_mlp_decoder()
        else:
            model = SACRED(config)
        
        return model.to(self.device)
    
    def _create_random_encoder(self):
        """Create random embedding encoder"""
        return nn.Sequential(
            nn.Embedding(1000, 384),
            nn.LayerNorm(384)
        )
    
    def _create_mlp_encoder(self):
        """Create MLP encoder for scaffold"""
        return nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384)
        )
    
    def _create_simple_property_encoder(self):
        """Create property encoder without attention"""
        return nn.Sequential(
            nn.Linear(13, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 384)
        )
    
    def _create_concat_fusion(self):
        """Create simple concatenation fusion"""
        class ConcatFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(384 * 3, 512)
            
            def forward(self, mol_emb, scaffold_emb, property_emb):
                concat = torch.cat([mol_emb, scaffold_emb, property_emb], dim=-1)
                return self.projection(concat)
        
        return ConcatFusion()
    
    def _create_deterministic_model(self, config):
        """Create deterministic version without VAE"""
        class DeterministicSACRED(SACRED):
            def forward(self, smiles, scaffold_graphs, properties, target_smiles=None):
                # Same as SACRED but without VAE
                mol_emb = self.mol_encoder(smiles)
                mol_emb = self.mol_projection(mol_emb)
                
                scaffold_emb = self.scaffold_encoder(
                    scaffold_graphs.x,
                    scaffold_graphs.edge_index,
                    scaffold_graphs.batch
                )
                
                property_emb = self.property_encoder(properties)
                fused = self.fusion(mol_emb, scaffold_emb, property_emb)
                
                # Direct decoding without VAE
                output = self.decoder(fused, target_smiles)
                
                return {
                    'output': output,
                    'fused_representation': fused
                }
        
        return DeterministicSACRED(config)
    
    def _create_mlp_decoder(self):
        """Create simple MLP decoder"""
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )
    
    def run_ablation(self, test_data: List[Dict], num_samples: int = 100) -> pd.DataFrame:
        """Run ablation study on all configurations"""
        results = []
        
        for config_name, config_info in self.ablation_configs.items():
            logger.info(f"Testing configuration: {config_info['description']}")
            
            # Create model
            model = self.create_ablated_model(config_name)
            
            # Load pretrained weights if available
            checkpoint_path = Path(f"checkpoints/{config_name}_best.pt")
            if checkpoint_path.exists():
                model.load_state_dict(torch.load(checkpoint_path))
            
            # Evaluate
            metrics = self.evaluate_model(model, test_data[:num_samples])
            
            # Store results
            result = {
                'Configuration': config_name,
                'Description': config_info['description'],
                **metrics
            }
            results.append(result)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate relative performance
        full_model_metrics = df[df['Configuration'] == 'full_model'].iloc[0]
        for col in df.columns:
            if col not in ['Configuration', 'Description']:
                df[f'{col}_relative'] = df[col] / full_model_metrics[col]
        
        return df
    
    def evaluate_model(self, model: SACRED, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a single model configuration"""
        model.eval()
        
        all_generated = []
        all_scaffolds = []
        all_constraints = []
        
        with torch.no_grad():
            for batch in self._create_batches(test_data, batch_size=16):
                # Prepare batch
                scaffold_graphs = batch['scaffold_graphs'].to(self.device)
                properties = batch['properties'].to(self.device)
                
                # Generate molecules
                generated = model.generate(
                    scaffold_graphs=scaffold_graphs,
                    properties=properties,
                    num_samples=5,
                    temperature=0.8
                )
                
                # Decode and store
                tokenizer = SMILESTokenizer()
                for gen_batch in generated:
                    for tokens in gen_batch:
                        smiles = tokenizer.decode(tokens.cpu().numpy())
                        all_generated.append(smiles)
                
                all_scaffolds.extend(batch['target_scaffolds'])
                all_constraints.extend(batch['property_constraints'])
        
        # Evaluate metrics
        metrics = self.evaluator.evaluate_batch(
            all_generated,
            target_scaffolds=all_scaffolds,
            property_constraints=self._merge_constraints(all_constraints)
        )
        
        # Flatten nested metrics
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f'{key}_{sub_key}'] = sub_value
            else:
                flat_metrics[key] = value
        
        return flat_metrics
    
    def _create_batches(self, data: List[Dict], batch_size: int):
        """Create batches from data"""
        collator = DataCollator(
            MolecularFeaturizer(),
            ADMETCalculator(),
            SMILESTokenizer()
        )
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            yield collator.collate_batch(batch_data)
    
    def _merge_constraints(self, constraints_list: List[Dict]) -> Dict:
        """Merge list of constraints into single dict"""
        merged = {}
        for constraints in constraints_list:
            for prop, range_val in constraints.items():
                if prop not in merged:
                    merged[prop] = range_val
        return merged
    
    def visualize_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize ablation study results"""
        # Select key metrics
        key_metrics = [
            'validity', 'uniqueness', 'diversity',
            'scaffold_retention', 'property_satisfaction_overall'
        ]
        
        # Filter columns
        plot_data = results_df[['Configuration'] + key_metrics].set_index('Configuration')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            plot_data.T,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Score'}
        )
        plt.title('SACRED Ablation Study Results')
        plt.xlabel('Configuration')
        plt.ylabel('Metric')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        # Create bar plot for relative performance
        relative_cols = [col for col in results_df.columns if '_relative' in col]
        if relative_cols:
            plot_data_relative = results_df[['Configuration'] + relative_cols].set_index('Configuration')
            
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_data_relative.plot(kind='bar', ax=ax)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Relative Performance')
            ax.set_title('Component Contribution Analysis')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                save_path_relative = save_path.replace('.png', '_relative.png')
                plt.savefig(save_path_relative, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()


class ModelComparison:
    """Compare SACRED with baseline models"""
    
    def __init__(self):
        self.baselines = {
            'random': self.random_baseline,
            'scaffold_decorator': self.scaffold_decorator_baseline,
            'property_vae': self.property_vae_baseline,
            'junction_tree_vae': self.jtvae_baseline,
        }
        self.evaluator = ConstraintEvaluator()
    
    def random_baseline(self, constraints: Dict, num_samples: int = 100) -> List[str]:
        """Random SMILES generation baseline"""
        # Simple random SMILES generation
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br']
        generated = []
        
        for _ in range(num_samples):
            length = np.random.randint(10, 30)
            smiles = ''
            for _ in range(length):
                if np.random.random() < 0.8:
                    smiles += np.random.choice(elements)
                if np.random.random() < 0.2:
                    smiles += np.random.choice(['(', ')', '=', '#'])
            generated.append(smiles)
        
        return generated
    
    def scaffold_decorator_baseline(self, constraints: Dict, num_samples: int = 100) -> List[str]:
        """Scaffold decoration baseline"""
        # Simplified scaffold decoration
        scaffold = constraints.get('scaffold', 'c1ccccc1')
        decorations = ['C', 'CC', 'CCC', 'N', 'O', 'F', 'Cl']
        
        generated = []
        for _ in range(num_samples):
            decorated = scaffold
            num_decorations = np.random.randint(1, 4)
            for _ in range(num_decorations):
                decoration = np.random.choice(decorations)
                decorated = f"({decoration}){decorated}"
            generated.append(decorated)
        
        return generated
    
    def property_vae_baseline(self, constraints: Dict, num_samples: int = 100) -> List[str]:
        """Property-conditioned VAE baseline (simplified)"""
        # This would normally load a pretrained VAE
        # For now, return scaffold with modifications
        scaffold = constraints.get('scaffold', 'c1ccccc1')
        
        generated = []
        for _ in range(num_samples):
            # Add functional groups based on properties
            modified = scaffold
            if constraints.get('LogP', (0, 5))[1] > 3:
                modified += 'CCCC'  # Add lipophilic groups
            if constraints.get('HBD', (0, 5))[1] > 2:
                modified += 'N'  # Add H-bond donors
            generated.append(modified)
        
        return generated
    
    def jtvae_baseline(self, constraints: Dict, num_samples: int = 100) -> List[str]:
        """Junction Tree VAE baseline (simplified)"""
        # Simplified version - would normally use actual JT-VAE
        templates = [
            'c1ccccc1CC',
            'c1ccc(cc1)N',
            'c1cccnc1',
            'C1CCCCC1'
        ]
        
        generated = []
        for _ in range(num_samples):
            template = np.random.choice(templates)
            # Modify based on constraints
            if np.random.random() < 0.5:
                template += 'O'
            if np.random.random() < 0.3:
                template += 'N'
            generated.append(template)
        
        return generated
    
    def compare_all(self, test_constraints: List[Dict], 
                   sacred_model: Optional[SACRED] = None) -> pd.DataFrame:
        """Compare all models on test constraints"""
        results = []
        
        for constraint_set in tqdm(test_constraints, desc="Testing constraints"):
            # Baseline models
            for baseline_name, baseline_func in self.baselines.items():
                generated = baseline_func(constraint_set, num_samples=100)
                
                metrics = self.evaluator.evaluate_batch(
                    generated,
                    target_scaffolds=[constraint_set.get('scaffold')] * len(generated),
                    property_constraints=constraint_set.get('properties')
                )
                
                result = {
                    'Model': baseline_name,
                    'Constraint_ID': constraint_set.get('id', 'unknown'),
                    **self._flatten_metrics(metrics)
                }
                results.append(result)
            
            # SACRED model
            if sacred_model is not None:
                generated = self._generate_sacred(sacred_model, constraint_set)
                
                metrics = self.evaluator.evaluate_batch(
                    generated,
                    target_scaffolds=[constraint_set.get('scaffold')] * len(generated),
                    property_constraints=constraint_set.get('properties')
                )
                
                result = {
                    'Model': 'SACRED',
                    'Constraint_ID': constraint_set.get('id', 'unknown'),
                    **self._flatten_metrics(metrics)
                }
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Aggregate by model
        aggregated = df.groupby('Model').mean().drop('Constraint_ID', axis=1, errors='ignore')
        
        return aggregated
    
    def _flatten_metrics(self, metrics: Dict) -> Dict:
        """Flatten nested metrics dictionary"""
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat[f'{key}_{sub_key}'] = sub_value
            elif isinstance(value, (int, float)):
                flat[key] = value
        return flat
    
    def _generate_sacred(self, model: SACRED, constraints: Dict) -> List[str]:
        """Generate molecules using SACRED model"""
        # Prepare inputs
        featurizer = MolecularFeaturizer()
        admet_calc = ADMETCalculator()
        tokenizer = SMILESTokenizer()
        
        # Create scaffold graph
        scaffold_graph = featurizer.smiles_to_graph(constraints.get('scaffold', 'c1ccccc1'))
        scaffold_batch = Batch.from_data_list([scaffold_graph])
        
        # Create property tensor
        properties = []
        for prop_name in admet_calc.PROPERTY_FUNCTIONS.keys():
            if prop_name in constraints.get('properties', {}):
                min_val, max_val = constraints['properties'][prop_name]
                properties.append((min_val + max_val) / 2)  # Use midpoint
            else:
                properties.append(0.5)  # Default
        
        properties_tensor = torch.tensor([properties], dtype=torch.float32)
        
        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                scaffold_graphs=scaffold_batch.to(model.device),
                properties=properties_tensor.to(model.device),
                num_samples=100,
                temperature=0.8
            )
        
        # Decode
        generated_smiles = []
        for tokens in generated_tokens[0]:  # First batch
            smiles = tokenizer.decode(tokens.cpu().numpy())
            generated_smiles.append(smiles)
        
        return generated_smiles


def main():
    """Run complete experimental validation"""
    
    # Configuration
    config = {
        'mol_dim': 384,
        'scaffold_dim': 384,
        'property_dim': 384,
        'fusion_dim': 512,
        'latent_dim': 256,
        'vocab_size': 100,
        'freeze_chemberta': True
    }
    
    # Load test data
    test_data = []  # Would load from file
    test_constraints = [
        {
            'id': 'test_1',
            'scaffold': 'c1ccccc1',
            'properties': {
                'MW': (200, 400),
                'LogP': (1, 3),
                'HBA': (2, 5)
            }
        }
    ]
    
    # Run ablation study
    logger.info("Starting ablation study...")
    ablation = AblationStudy(config)
    ablation_results = ablation.run_ablation(test_data, num_samples=50)
    
    # Save results
    ablation_results.to_csv('results/ablation_study.csv', index=False)
    ablation.visualize_results(ablation_results, 'results/ablation_study.png')
    
    # Model comparison
    logger.info("Starting model comparison...")
    comparison = ModelComparison()
    
    # Load SACRED model
    sacred_model = SACRED(config)
    # sacred_model.load_state_dict(torch.load('checkpoints/sacred_best.pt'))
    
    comparison_results = comparison.compare_all(test_constraints, sacred_model)
    comparison_results.to_csv('results/model_comparison.csv')
    
    logger.info("Experimental validation complete!")
    
    # Print summary
    print("\n=== Ablation Study Results ===")
    print(ablation_results[['Configuration', 'validity', 'scaffold_retention', 
                           'property_satisfaction_overall']].to_string())
    
    print("\n=== Model Comparison Results ===")
    print(comparison_results[['validity', 'uniqueness', 'diversity']].to_string())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()