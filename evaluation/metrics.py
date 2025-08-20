"""
Comprehensive evaluation metrics for SACRED model
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MolecularMetrics:
    """Comprehensive metrics for molecular generation evaluation"""
    
    @staticmethod
    def validity(smiles_list: List[str]) -> float:
        """Calculate fraction of valid SMILES"""
        valid_count = 0
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
            except:
                pass
        
        return valid_count / len(smiles_list) if smiles_list else 0.0
    
    @staticmethod
    def uniqueness(smiles_list: List[str]) -> float:
        """Calculate fraction of unique valid molecules"""
        valid_canonical = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    valid_canonical.append(canonical)
            except:
                pass
        
        if not valid_canonical:
            return 0.0
        
        return len(set(valid_canonical)) / len(valid_canonical)
    
    @staticmethod
    def novelty(generated_smiles: List[str], reference_smiles: List[str]) -> float:
        """Calculate fraction of novel molecules not in reference set"""
        reference_canonical = set()
        for smiles in reference_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    reference_canonical.add(canonical)
            except:
                pass
        
        novel_count = 0
        valid_count = 0
        
        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    valid_count += 1
                    if canonical not in reference_canonical:
                        novel_count += 1
            except:
                pass
        
        return novel_count / valid_count if valid_count > 0 else 0.0
    
    @staticmethod
    def diversity(smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> float:
        """Calculate internal diversity using Tanimoto distance"""
        fingerprints = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
                    fingerprints.append(fp)
            except:
                pass
        
        if len(fingerprints) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        return 1 - np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def scaffold_retention(generated_smiles: List[str], target_scaffold: str) -> float:
        """Calculate fraction of molecules retaining target scaffold"""
        target_mol = Chem.MolFromSmiles(target_scaffold)
        if target_mol is None:
            return 0.0
        
        retained_count = 0
        valid_count = 0
        
        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    # Check if target scaffold is substructure
                    if mol.HasSubstructMatch(target_mol):
                        retained_count += 1
                    else:
                        # Also check Murcko scaffold similarity
                        gen_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        gen_scaffold_smiles = Chem.MolToSmiles(gen_scaffold)
                        
                        if gen_scaffold_smiles == target_scaffold:
                            retained_count += 1
            except:
                pass
        
        return retained_count / valid_count if valid_count > 0 else 0.0
    
    @staticmethod
    def property_satisfaction(smiles_list: List[str], 
                            constraints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate satisfaction rate for each property constraint"""
        
        property_functions = {
            'MW': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'HBA': Descriptors.NumHAcceptors,
            'HBD': Descriptors.NumHDonors,
            'TPSA': Descriptors.TPSA,
            'QED': Descriptors.qed,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
        }
        
        satisfaction_counts = {prop: 0 for prop in constraints}
        valid_count = 0
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                valid_count += 1
                
                for prop, (min_val, max_val) in constraints.items():
                    if prop in property_functions:
                        value = property_functions[prop](mol)
                        if min_val <= value <= max_val:
                            satisfaction_counts[prop] += 1
            except:
                pass
        
        if valid_count == 0:
            return {prop: 0.0 for prop in constraints}
        
        satisfaction_rates = {
            prop: count / valid_count 
            for prop, count in satisfaction_counts.items()
        }
        
        # Overall satisfaction (all constraints met)
        all_satisfied_count = 0
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                all_met = True
                for prop, (min_val, max_val) in constraints.items():
                    if prop in property_functions:
                        value = property_functions[prop](mol)
                        if not (min_val <= value <= max_val):
                            all_met = False
                            break
                
                if all_met:
                    all_satisfied_count += 1
            except:
                pass
        
        satisfaction_rates['overall'] = all_satisfied_count / valid_count if valid_count > 0 else 0.0
        
        return satisfaction_rates
    
    @staticmethod
    def synthetic_accessibility(smiles_list: List[str]) -> Dict[str, float]:
        """Calculate synthetic accessibility scores"""
        sa_scores = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Simplified SA score calculation
                # (Full implementation would use sascorer)
                score = 0
                
                # Penalize large molecules
                if mol.GetNumAtoms() > 50:
                    score += 2
                elif mol.GetNumAtoms() > 30:
                    score += 1
                
                # Penalize many rings
                ring_count = Descriptors.RingCount(mol)
                if ring_count > 4:
                    score += 2
                elif ring_count > 3:
                    score += 1
                
                # Penalize stereocenters
                stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                score += len(stereo_centers) * 0.5
                
                # Normalize to 1-10 scale
                sa_score = min(10, max(1, 1 + score))
                sa_scores.append(sa_score)
                
            except:
                pass
        
        if not sa_scores:
            return {'mean': 0.0, 'std': 0.0, 'good_fraction': 0.0}
        
        sa_array = np.array(sa_scores)
        
        return {
            'mean': float(np.mean(sa_array)),
            'std': float(np.std(sa_array)),
            'good_fraction': float(np.sum(sa_array <= 3.5) / len(sa_array))  # SA <= 3.5 is good
        }
    
    @staticmethod
    def distribution_matching(generated_properties: np.ndarray, 
                            reference_properties: np.ndarray) -> Dict[str, float]:
        """Calculate distribution matching metrics"""
        
        metrics = {}
        
        # Wasserstein distance for each property
        for i in range(generated_properties.shape[1]):
            gen_dist = generated_properties[:, i]
            ref_dist = reference_properties[:, i]
            
            # Normalize to same scale
            gen_dist = (gen_dist - gen_dist.min()) / (gen_dist.max() - gen_dist.min() + 1e-8)
            ref_dist = (ref_dist - ref_dist.min()) / (ref_dist.max() - ref_dist.min() + 1e-8)
            
            # Wasserstein distance
            wasserstein = stats.wasserstein_distance(gen_dist, ref_dist)
            metrics[f'wasserstein_prop_{i}'] = float(wasserstein)
            
            # KL divergence (after binning)
            hist_gen, bins = np.histogram(gen_dist, bins=20, density=True)
            hist_ref, _ = np.histogram(ref_dist, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            hist_gen = hist_gen + 1e-10
            hist_ref = hist_ref + 1e-10
            
            kl_div = stats.entropy(hist_gen, hist_ref)
            metrics[f'kl_divergence_prop_{i}'] = float(kl_div)
        
        # Overall metrics
        metrics['mean_wasserstein'] = np.mean([v for k, v in metrics.items() if 'wasserstein' in k])
        metrics['mean_kl_divergence'] = np.mean([v for k, v in metrics.items() if 'kl_divergence' in k])
        
        return metrics


class ConstraintEvaluator:
    """Evaluate constraint satisfaction for generated molecules"""
    
    def __init__(self):
        self.metrics = MolecularMetrics()
    
    def evaluate_batch(self, 
                       generated_smiles: List[str],
                       target_scaffolds: Optional[List[str]] = None,
                       property_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
                       reference_smiles: Optional[List[str]] = None) -> Dict[str, any]:
        """Comprehensive evaluation of a batch of generated molecules"""
        
        results = {
            'validity': self.metrics.validity(generated_smiles),
            'uniqueness': self.metrics.uniqueness(generated_smiles),
            'diversity': self.metrics.diversity(generated_smiles),
        }
        
        # Novelty (if reference set provided)
        if reference_smiles:
            results['novelty'] = self.metrics.novelty(generated_smiles, reference_smiles)
        
        # Scaffold retention (if target scaffolds provided)
        if target_scaffolds:
            scaffold_retentions = []
            for gen_smiles, target_scaffold in zip(generated_smiles, target_scaffolds):
                retention = self.metrics.scaffold_retention([gen_smiles], target_scaffold)
                scaffold_retentions.append(retention)
            results['scaffold_retention'] = np.mean(scaffold_retentions)
        
        # Property satisfaction (if constraints provided)
        if property_constraints:
            satisfaction = self.metrics.property_satisfaction(generated_smiles, property_constraints)
            results['property_satisfaction'] = satisfaction
        
        # Synthetic accessibility
        results['synthetic_accessibility'] = self.metrics.synthetic_accessibility(generated_smiles)
        
        return results
    
    def compare_models(self, 
                       model_results: Dict[str, List[str]],
                       constraints: Dict) -> pd.DataFrame:
        """Compare multiple models on the same constraints"""
        
        comparison_data = []
        
        for model_name, generated_smiles in model_results.items():
            metrics = self.evaluate_batch(
                generated_smiles,
                target_scaffolds=constraints.get('scaffolds'),
                property_constraints=constraints.get('properties'),
                reference_smiles=constraints.get('reference')
            )
            
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate aggregate scores
        df['Aggregate_Score'] = (
            df['validity'] * 0.2 +
            df['uniqueness'] * 0.15 +
            df['diversity'] * 0.15 +
            df.get('scaffold_retention', 0) * 0.25 +
            df.get('property_satisfaction', {}).get('overall', 0) * 0.25
        )
        
        return df.sort_values('Aggregate_Score', ascending=False)


class PerformanceTracker:
    """Track model performance during training"""
    
    def __init__(self, metrics_to_track: List[str] = None):
        self.metrics_to_track = metrics_to_track or [
            'validity', 'uniqueness', 'diversity', 
            'scaffold_retention', 'property_satisfaction'
        ]
        self.history = {metric: [] for metric in self.metrics_to_track}
        self.best_scores = {metric: 0.0 for metric in self.metrics_to_track}
        self.best_epoch = {metric: 0 for metric in self.metrics_to_track}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update tracking with new metrics"""
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, dict):
                    value = value.get('overall', 0.0)
                
                self.history[metric].append(value)
                
                if value > self.best_scores[metric]:
                    self.best_scores[metric] = value
                    self.best_epoch[metric] = epoch
    
    def get_best(self) -> Dict[str, Tuple[float, int]]:
        """Get best scores and their epochs"""
        return {
            metric: (self.best_scores[metric], self.best_epoch[metric])
            for metric in self.metrics_to_track
        }
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(
            len(self.metrics_to_track), 1, 
            figsize=(10, 4 * len(self.metrics_to_track))
        )
        
        if len(self.metrics_to_track) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, self.metrics_to_track):
            ax.plot(self.history[metric])
            ax.set_title(f'{metric} over epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True)
            
            # Mark best epoch
            best_epoch = self.best_epoch[metric]
            best_score = self.best_scores[metric]
            ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
            ax.text(best_epoch, best_score, f'Best: {best_score:.3f}', 
                   ha='right', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()