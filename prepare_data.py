#!/usr/bin/env python3
"""
Prepare training data for SACRED model
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(output_dir: str, num_samples: int = 1000):
    """Create sample training data for testing"""
    
    # Sample SMILES from various drug-like molecules
    sample_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C(=O)N(C2=CC=CC=C12)C3=CC=C(C=C3)CC(=O)O)C4=CC=CC=C4Cl",  # Indomethacin
        "COc1ccc2nc(S(N)(=O)=O)sc2c1",  # Sulfamethoxazole
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib
        "CC(C)NCC(COc1cccc2c1cccc2)O",  # Propranolol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1CCC(CC1)C2=CC=C(C=C2)Cl",  # Chlorphenamine
        "COC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)O",  # Daidzein
        "C1CCN(CC1)CCOC2=CC=CC=C2",  # Phenoxybenzamine
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # Theophylline
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F",  # Celecoxib
        "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",  # Omeprazole
        "CN1CCN(CC1)C2=NC3=C(C=CC=C3N2C4=CC=C(C=C4)F)Cl",  # Loratadine
    ]
    
    # Create variations by adding functional groups
    variations = []
    for base_smiles in sample_smiles:
        mol = Chem.MolFromSmiles(base_smiles)
        if mol is None:
            continue
        
        # Original molecule
        variations.append(base_smiles)
        
        # Create variations
        for _ in range(num_samples // len(sample_smiles)):
            # Random modifications
            modified = base_smiles
            
            # Add methyl groups
            if np.random.random() < 0.3:
                modified = modified.replace('C', 'C(C)', 1)
            
            # Add hydroxyl
            if np.random.random() < 0.3:
                modified = modified.replace('C', 'C(O)', 1)
            
            # Add amine
            if np.random.random() < 0.2:
                modified = modified.replace('C', 'C(N)', 1)
            
            # Validate modified SMILES
            mol = Chem.MolFromSmiles(modified)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                variations.append(canonical)
    
    # Create dataset
    data = []
    for smiles in variations[:num_samples]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # Calculate basic properties
        properties = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'QED': Descriptors.qed(mol),
        }
        
        data.append({
            'smiles': smiles,
            'properties': properties
        })
    
    # Split into train/val/test
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Save data
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save as JSONL
    with open(output_path / 'train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(output_path / 'val.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open(output_path / 'test.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created dataset with {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    logger.info(f"Saved to {output_path}")


def process_chembl_data(input_file: str, output_dir: str, max_samples: int = None):
    """Process ChEMBL or other molecular datasets"""
    
    logger.info(f"Processing {input_file}...")
    
    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.sdf'):
        from rdkit.Chem import PandasTools
        df = PandasTools.LoadSDF(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    # Get SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'canonical_smiles', 'Smiles']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError("No SMILES column found in data")
    
    # Process molecules
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if max_samples and idx >= max_samples:
            break
        
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            continue
        
        # Filter by drug-likeness
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        if mw < 150 or mw > 500:
            continue
        if logp < -2 or logp > 5:
            continue
        
        # Calculate properties
        properties = {
            'MW': mw,
            'LogP': logp,
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'QED': Descriptors.qed(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        }
        
        processed_data.append({
            'smiles': Chem.MolToSmiles(mol),
            'properties': properties,
            'chembl_id': row.get('chembl_id', f'MOL_{idx}')
        })
    
    logger.info(f"Processed {len(processed_data)} valid molecules")
    
    # Split and save
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save as JSONL
    with open(output_path / 'train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(output_path / 'val.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open(output_path / 'test.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SACRED training")
    parser.add_argument('--mode', type=str, choices=['sample', 'chembl'], default='sample',
                       help='Data preparation mode')
    parser.add_argument('--input', type=str, help='Input file for ChEMBL mode')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of samples for sample mode')
    parser.add_argument('--max_samples', type=int, help='Max samples for ChEMBL mode')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        create_sample_data(args.output, args.num_samples)
    elif args.mode == 'chembl':
        if not args.input:
            raise ValueError("Input file required for ChEMBL mode")
        process_chembl_data(args.input, args.output, args.max_samples)
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()