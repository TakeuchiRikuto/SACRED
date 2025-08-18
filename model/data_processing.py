"""
Data processing and constraint handling for SACRED
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """Convert molecules to graph representations"""
    
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-2, -1, 0, 1, 2],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    }
    
    def __init__(self):
        self.atom_feature_size = self._calculate_feature_size()
    
    def _calculate_feature_size(self) -> int:
        return (
            len(self.ATOM_FEATURES['atomic_num']) +
            len(self.ATOM_FEATURES['degree']) +
            len(self.ATOM_FEATURES['formal_charge']) +
            len(self.ATOM_FEATURES['chiral_tag']) +
            len(self.ATOM_FEATURES['num_Hs']) +
            len(self.ATOM_FEATURES['hybridization']) +
            2  # is_aromatic, is_in_ring
        )
    
    def atom_to_features(self, atom: Chem.Atom) -> np.ndarray:
        features = []
        
        # One-hot encoding for categorical features
        features.extend(self._one_hot(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']))
        features.extend(self._one_hot(atom.GetDegree(), self.ATOM_FEATURES['degree']))
        features.extend(self._one_hot(atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge']))
        features.extend(self._one_hot(atom.GetChiralTag(), self.ATOM_FEATURES['chiral_tag']))
        features.extend(self._one_hot(atom.GetTotalNumHs(), self.ATOM_FEATURES['num_Hs']))
        features.extend(self._one_hot(atom.GetHybridization(), self.ATOM_FEATURES['hybridization']))
        
        # Binary features
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(1 if atom.IsInRing() else 0)
        
        return np.array(features, dtype=np.float32)
    
    def _one_hot(self, value: Union[int, object], allowable_set: List) -> List[int]:
        if value not in allowable_set:
            value = allowable_set[-1] if allowable_set else 0
        return [1 if v == value else 0 for v in allowable_set]
    
    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        """Convert RDKit molecule to PyTorch Geometric Data object"""
        
        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self.atom_to_features(atom))
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add edges in both directions
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond features (simplified)
            bond_type = bond.GetBondTypeAsDouble()
            edge_features.extend([bond_type, bond_type])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to graph"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.mol_to_graph(mol)


class ADMETCalculator:
    """Calculate ADMET properties for molecules"""
    
    PROPERTY_FUNCTIONS = {
        'MW': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'HBA': Descriptors.NumHAcceptors,
        'HBD': Descriptors.NumHDonors,
        'TPSA': Descriptors.TPSA,
        'QED': Descriptors.qed,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumHeteroatoms': Descriptors.NumHeteroatoms,
        'NumHeavyAtoms': Descriptors.HeavyAtomCount,
        'MolMR': Crippen.MolMR,
        'BertzCT': Descriptors.BertzCT,
        'FractionCSP3': Descriptors.FractionCSP3 if hasattr(Descriptors, 'FractionCSP3') else lambda x: 0.5
    }
    
    PROPERTY_RANGES = {
        'MW': (150, 500),
        'LogP': (-0.4, 5.6),
        'HBA': (0, 10),
        'HBD': (0, 5),
        'TPSA': (0, 140),
        'QED': (0, 1),
        'NumRotatableBonds': (0, 10),
        'NumAromaticRings': (0, 4),
        'NumHeteroatoms': (1, 10),
        'NumHeavyAtoms': (10, 50),
        'MolMR': (40, 130),
        'BertzCT': (0, 2000),
        'FractionCSP3': (0, 1)
    }
    
    def calculate_properties(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate normalized ADMET properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = []
        for prop_name, func in self.PROPERTY_FUNCTIONS.items():
            try:
                value = func(mol)
                # Normalize to [0, 1]
                min_val, max_val = self.PROPERTY_RANGES[prop_name]
                normalized = (value - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
                properties.append(normalized)
            except Exception as e:
                logger.warning(f"Error calculating {prop_name}: {e}")
                properties.append(0.5)  # Default to middle value
        
        return np.array(properties, dtype=np.float32)
    
    def check_constraints(self, smiles: str, constraints: Dict[str, Tuple[float, float]]) -> Tuple[bool, Dict]:
        """Check if molecule satisfies property constraints"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {}
        
        results = {}
        all_satisfied = True
        
        for prop_name, (min_val, max_val) in constraints.items():
            if prop_name in self.PROPERTY_FUNCTIONS:
                value = self.PROPERTY_FUNCTIONS[prop_name](mol)
                satisfied = min_val <= value <= max_val
                results[prop_name] = {
                    'value': value,
                    'satisfied': satisfied,
                    'range': (min_val, max_val)
                }
                if not satisfied:
                    all_satisfied = False
        
        return all_satisfied, results


class ScaffoldExtractor:
    """Extract and compare molecular scaffolds"""
    
    @staticmethod
    def extract_scaffold(smiles: str) -> Optional[str]:
        """Extract Murcko scaffold from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception as e:
            logger.warning(f"Error extracting scaffold: {e}")
            return None
    
    @staticmethod
    def calculate_scaffold_similarity(smiles1: str, smiles2: str) -> float:
        """Calculate Tanimoto similarity between scaffolds"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    @staticmethod
    def check_scaffold_constraint(generated_smiles: str, target_scaffold: str, 
                                 threshold: float = 0.7) -> bool:
        """Check if generated molecule contains target scaffold"""
        generated_scaffold = ScaffoldExtractor.extract_scaffold(generated_smiles)
        if generated_scaffold is None:
            return False
        
        similarity = ScaffoldExtractor.calculate_scaffold_similarity(
            generated_scaffold, target_scaffold
        )
        
        return similarity >= threshold


class ConstraintSampler:
    """Sample property constraints for training"""
    
    def __init__(self, property_ranges: Optional[Dict] = None):
        self.property_ranges = property_ranges or ADMETCalculator.PROPERTY_RANGES
    
    def sample_constraints(self, 
                          num_constraints: int = 3,
                          difficulty: str = 'medium') -> Dict[str, Tuple[float, float]]:
        """Sample random property constraints"""
        
        # Select random properties
        selected_props = np.random.choice(
            list(self.property_ranges.keys()),
            size=min(num_constraints, len(self.property_ranges)),
            replace=False
        )
        
        constraints = {}
        for prop in selected_props:
            min_val, max_val = self.property_ranges[prop]
            
            if difficulty == 'easy':
                # Wide range (60% of full range)
                range_width = (max_val - min_val) * 0.6
            elif difficulty == 'medium':
                # Medium range (40% of full range)
                range_width = (max_val - min_val) * 0.4
            else:  # hard
                # Narrow range (20% of full range)
                range_width = (max_val - min_val) * 0.2
            
            # Random center point
            center = np.random.uniform(
                min_val + range_width/2,
                max_val - range_width/2
            )
            
            constraints[prop] = (
                max(min_val, center - range_width/2),
                min(max_val, center + range_width/2)
            )
        
        return constraints


class DataCollator:
    """Collate batch data for SACRED model"""
    
    def __init__(self, featurizer: MolecularFeaturizer, 
                 admet_calculator: ADMETCalculator,
                 tokenizer=None):
        self.featurizer = featurizer
        self.admet_calculator = admet_calculator
        self.tokenizer = tokenizer
    
    def collate_batch(self, batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List, Batch]]:
        """Collate batch of molecular data"""
        
        smiles_list = []
        scaffold_graphs = []
        properties_list = []
        target_tokens = []
        
        for item in batch:
            # SMILES
            smiles_list.append(item['smiles'])
            
            # Scaffold graph
            scaffold_smiles = item.get('scaffold', '')
            if scaffold_smiles:
                graph = self.featurizer.smiles_to_graph(scaffold_smiles)
                if graph is not None:
                    scaffold_graphs.append(graph)
            
            # Properties
            properties = item.get('properties')
            if properties is None:
                properties = self.admet_calculator.calculate_properties(item['smiles'])
            properties_list.append(properties)
            
            # Target tokens (if tokenizer provided)
            if self.tokenizer and 'target_smiles' in item:
                tokens = self.tokenizer.encode(item['target_smiles'])
                target_tokens.append(tokens)
        
        # Batch graphs
        scaffold_batch = Batch.from_data_list(scaffold_graphs) if scaffold_graphs else None
        
        # Stack properties
        properties_tensor = torch.tensor(np.array(properties_list), dtype=torch.float32)
        
        # Prepare target tokens
        if target_tokens:
            max_len = max(len(t) for t in target_tokens)
            padded_targets = []
            for tokens in target_tokens:
                padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
                padded_targets.append(padded)
            target_tensor = torch.tensor(padded_targets, dtype=torch.long)
        else:
            target_tensor = None
        
        return {
            'smiles': smiles_list,
            'scaffold_graphs': scaffold_batch,
            'properties': properties_tensor,
            'target_tokens': target_tensor
        }


class SMILESTokenizer:
    """Simple SMILES tokenizer"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        self.special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        # Basic SMILES tokens
        self.tokens = list('CNOPSFClBrI()[]=#-+\\/@')
        self.tokens.extend(['c', 'n', 'o', 's', 'p'])  # Aromatic atoms
        self.tokens.extend([str(i) for i in range(10)])  # Numbers
        
        # Build vocabulary
        self.vocab = {**self.special_tokens}
        for i, token in enumerate(self.tokens):
            self.vocab[token] = len(self.special_tokens) + i
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.special_tokens['<pad>']
        
    def encode(self, smiles: str) -> List[int]:
        """Encode SMILES to token IDs"""
        tokens = [self.special_tokens['<bos>']]
        
        i = 0
        while i < len(smiles):
            # Try two-character tokens first (Br, Cl)
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.vocab:
                    tokens.append(self.vocab[two_char])
                    i += 2
                    continue
            
            # Single character
            char = smiles[i]
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.special_tokens['<unk>'])
            i += 1
        
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to SMILES"""
        smiles = ''
        for tid in token_ids:
            if tid in self.inv_vocab:
                token = self.inv_vocab[tid]
                if token not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    smiles += token
        return smiles