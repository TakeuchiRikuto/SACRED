"""
Simplified data processing without DeepChem dependency
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Simplified version without RDKit dependency for testing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available, using mock functions")
    RDKIT_AVAILABLE = False


class SimpleMolecularFeaturizer:
    """Simplified molecular featurizer"""
    
    def __init__(self):
        self.atom_feature_size = 74  # Fixed size
    
    def smiles_to_graph(self, smiles: str) -> Optional[Dict]:
        """Convert SMILES to graph representation"""
        if not RDKIT_AVAILABLE:
            # Return dummy graph for testing
            return {
                'x': torch.randn(5, self.atom_feature_size),
                'edge_index': torch.tensor([[0,1,1,2,2,3,3,4], 
                                           [1,0,2,1,3,2,4,3]], dtype=torch.long)
            }
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Simple atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
            ]
            # Pad to fixed size
            features.extend([0] * (self.atom_feature_size - len(features)))
            atom_features.append(features)
        
        # Edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        
        if not edge_indices:
            edge_indices = [[0, 0]]  # Self-loop for single atom
        
        return {
            'x': torch.tensor(atom_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_indices, dtype=torch.long).t()
        }


class SimpleADMETCalculator:
    """Simplified ADMET calculator without DeepChem"""
    
    def __init__(self):
        self.num_properties = 13
    
    def calculate_properties(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate molecular properties"""
        if not RDKIT_AVAILABLE:
            # Return random properties for testing
            return np.random.rand(self.num_properties)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate basic properties
            properties = [
                Descriptors.MolWt(mol) / 500,  # Normalize
                Descriptors.MolLogP(mol) / 5,
                Descriptors.NumHAcceptors(mol) / 10,
                Descriptors.NumHDonors(mol) / 5,
                Descriptors.TPSA(mol) / 140,
                Descriptors.NumRotatableBonds(mol) / 10,
                Descriptors.NumAromaticRings(mol) / 4,
                Descriptors.NumHeteroatoms(mol) / 10,
                Descriptors.RingCount(mol) / 6,
                Descriptors.FractionCsp3(mol),
                Descriptors.NumSaturatedRings(mol) / 4,
                Descriptors.NumAliphaticRings(mol) / 4,
                0.5,  # Placeholder for QED
            ]
            
            # Clip to [0, 1]
            properties = np.clip(properties, 0, 1)
            return np.array(properties, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error calculating properties: {e}")
            return np.ones(self.num_properties) * 0.5


class SimpleScaffoldExtractor:
    """Simplified scaffold extraction"""
    
    @staticmethod
    def extract_scaffold(smiles: str) -> Optional[str]:
        """Extract scaffold from SMILES"""
        if not RDKIT_AVAILABLE:
            return "c1ccccc1"  # Return benzene as default
        
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception as e:
            logger.warning(f"Error extracting scaffold: {e}")
            return "c1ccccc1"
    
    @staticmethod
    def calculate_scaffold_similarity(smiles1: str, smiles2: str) -> float:
        """Calculate similarity between scaffolds"""
        if not RDKIT_AVAILABLE:
            return np.random.rand()
        
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            from rdkit import DataStructs
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.5


class SimpleSMILESTokenizer:
    """Simple SMILES tokenizer"""
    
    def __init__(self):
        # Basic SMILES tokens
        self.special_tokens = {
            '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
        }
        
        # Common SMILES characters
        chars = list('CNOSFPClBrI()[]=#-+123456789\\/@')
        chars.extend(['c', 'n', 'o', 's', 'p'])  # Aromatic
        
        self.vocab = {**self.special_tokens}
        for i, char in enumerate(chars):
            self.vocab[char] = len(self.special_tokens) + i
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.special_tokens['<pad>']
        self.vocab_size = len(self.vocab)
    
    def encode(self, smiles: str) -> List[int]:
        """Encode SMILES string to token IDs"""
        tokens = [self.special_tokens['<bos>']]
        
        i = 0
        while i < len(smiles):
            # Try two-character tokens (Br, Cl)
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in ['Br', 'Cl']:
                    tokens.append(self.vocab.get(two_char, self.special_tokens['<unk>']))
                    i += 2
                    continue
            
            # Single character
            char = smiles[i]
            tokens.append(self.vocab.get(char, self.special_tokens['<unk>']))
            i += 1
        
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to SMILES string"""
        smiles = ''
        for tid in token_ids:
            if tid in self.inv_vocab:
                token = self.inv_vocab[tid]
                if token not in self.special_tokens:
                    smiles += token
        return smiles


class SimpleDataCollator:
    """Simple data collator for batching"""
    
    def __init__(self):
        self.featurizer = SimpleMolecularFeaturizer()
        self.admet_calc = SimpleADMETCalculator()
        self.tokenizer = SimpleSMILESTokenizer()
    
    def collate_batch(self, batch: List[Dict]) -> Dict:
        """Collate batch of data"""
        from torch_geometric.data import Data, Batch
        
        smiles_list = []
        scaffold_graphs = []
        properties_list = []
        
        for item in batch:
            smiles_list.append(item.get('smiles', 'CCO'))
            
            # Create scaffold graph
            scaffold = item.get('scaffold', 'c1ccccc1')
            graph_dict = self.featurizer.smiles_to_graph(scaffold)
            if graph_dict:
                graph = Data(x=graph_dict['x'], edge_index=graph_dict['edge_index'])
                scaffold_graphs.append(graph)
            
            # Get properties
            props = item.get('properties')
            if props is None:
                props = self.admet_calc.calculate_properties(item.get('smiles', 'CCO'))
            properties_list.append(props)
        
        # Batch graphs
        if scaffold_graphs:
            scaffold_batch = Batch.from_data_list(scaffold_graphs)
        else:
            # Create dummy batch
            dummy_graph = Data(
                x=torch.randn(5, self.featurizer.atom_feature_size),
                edge_index=torch.tensor([[0,1], [1,0]], dtype=torch.long)
            )
            scaffold_batch = Batch.from_data_list([dummy_graph] * len(batch))
        
        # Stack properties
        properties_tensor = torch.tensor(np.array(properties_list), dtype=torch.float32)
        
        # Tokenize target SMILES
        target_tokens = []
        for item in batch:
            target = item.get('target_smiles', item.get('smiles', 'CCO'))
            tokens = self.tokenizer.encode(target)
            target_tokens.append(tokens)
        
        # Pad tokens
        max_len = max(len(t) for t in target_tokens) if target_tokens else 10
        padded_tokens = []
        for tokens in target_tokens:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        target_tensor = torch.tensor(padded_tokens, dtype=torch.long)
        
        return {
            'smiles': smiles_list,
            'scaffold_graphs': scaffold_batch,
            'properties': properties_tensor,
            'target_tokens': target_tensor,
            'scaffold': [item.get('scaffold', 'c1ccccc1') for item in batch]
        }