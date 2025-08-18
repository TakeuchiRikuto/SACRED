#!/usr/bin/env python3
"""
Molecular generation script for SACRED model
"""

import torch
import argparse
import json
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

from model.sacred_model import SACRED
from model.data_processing import (
    MolecularFeaturizer, ADMETCalculator,
    ScaffoldExtractor, SMILESTokenizer
)
from evaluation.metrics import ConstraintEvaluator
from torch_geometric.data import Batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoleculeGenerator:
    """Generate molecules with SACRED"""
    
    def __init__(self, model_path: str, config: dict = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SACRED(config or {})
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.featurizer = MolecularFeaturizer()
        self.admet_calc = ADMETCalculator()
        self.tokenizer = SMILESTokenizer()
        self.evaluator = ConstraintEvaluator()
        
        logger.info(f"Model loaded from {model_path}")
    
    def generate(self, 
                 scaffold: str = None,
                 properties: dict = None,
                 num_molecules: int = 10,
                 temperature: float = 0.8):
        """Generate molecules with constraints"""
        
        # Default scaffold if not provided
        if scaffold is None:
            scaffold = 'c1ccccc1'  # Benzene ring
        
        # Create scaffold graph
        scaffold_graph = self.featurizer.smiles_to_graph(scaffold)
        if scaffold_graph is None:
            logger.error(f"Invalid scaffold: {scaffold}")
            return []
        
        scaffold_batch = Batch.from_data_list([scaffold_graph]).to(self.device)
        
        # Create property tensor
        if properties is None:
            # Default: middle of range for all properties
            property_values = [0.5] * 13
        else:
            property_values = []
            for prop_name in self.admet_calc.PROPERTY_FUNCTIONS.keys():
                if prop_name in properties:
                    min_val, max_val = properties[prop_name]
                    # Use midpoint of desired range
                    value = (min_val + max_val) / 2
                    # Normalize
                    range_min, range_max = self.admet_calc.PROPERTY_RANGES[prop_name]
                    normalized = (value - range_min) / (range_max - range_min)
                    property_values.append(normalized)
                else:
                    property_values.append(0.5)
        
        property_tensor = torch.tensor([property_values], dtype=torch.float32).to(self.device)
        
        # Generate molecules
        with torch.no_grad():
            generated_tokens = self.model.generate(
                scaffold_graphs=scaffold_batch,
                properties=property_tensor,
                num_samples=num_molecules,
                temperature=temperature
            )
        
        # Decode SMILES
        generated_smiles = []
        for tokens_batch in generated_tokens:
            tokens = tokens_batch[0]  # First (and only) item in batch
            smiles = self.tokenizer.decode(tokens.cpu().numpy())
            generated_smiles.append(smiles)
        
        return generated_smiles
    
    def evaluate_generated(self, smiles_list: list, scaffold: str = None, 
                          properties: dict = None):
        """Evaluate generated molecules"""
        
        results = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({
                    'smiles': smiles,
                    'valid': False
                })
                continue
            
            result = {
                'smiles': smiles,
                'valid': True,
                'canonical_smiles': Chem.MolToSmiles(mol)
            }
            
            # Check scaffold
            if scaffold:
                gen_scaffold = ScaffoldExtractor.extract_scaffold(smiles)
                similarity = ScaffoldExtractor.calculate_scaffold_similarity(
                    gen_scaffold, scaffold
                ) if gen_scaffold else 0.0
                result['scaffold_similarity'] = similarity
                result['scaffold_retained'] = similarity > 0.7
            
            # Calculate properties
            calc_props = self.admet_calc.calculate_properties(smiles)
            if calc_props is not None:
                # Denormalize properties
                for i, prop_name in enumerate(self.admet_calc.PROPERTY_FUNCTIONS.keys()):
                    range_min, range_max = self.admet_calc.PROPERTY_RANGES[prop_name]
                    value = calc_props[i] * (range_max - range_min) + range_min
                    result[prop_name] = value
                    
                    # Check constraint satisfaction
                    if properties and prop_name in properties:
                        min_val, max_val = properties[prop_name]
                        result[f'{prop_name}_satisfied'] = min_val <= value <= max_val
            
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate molecules with SACRED")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--scaffold', type=str, help='Scaffold SMILES')
    parser.add_argument('--properties', type=str, help='Property constraints as JSON')
    parser.add_argument('--num_molecules', type=int, default=10, help='Number to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--output', type=str, default='generated_molecules.csv', help='Output file')
    parser.add_argument('--visualize', action='store_true', help='Create molecular grid image')
    
    args = parser.parse_args()
    
    # Parse property constraints
    properties = None
    if args.properties:
        properties = json.loads(args.properties)
    
    # Initialize generator
    generator = MoleculeGenerator(args.model)
    
    # Generate molecules
    logger.info(f"Generating {args.num_molecules} molecules...")
    generated = generator.generate(
        scaffold=args.scaffold,
        properties=properties,
        num_molecules=args.num_molecules,
        temperature=args.temperature
    )
    
    logger.info(f"Generated {len(generated)} SMILES")
    
    # Evaluate
    results = generator.evaluate_generated(generated, args.scaffold, properties)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    valid_count = df['valid'].sum()
    print(f"\n=== Generation Summary ===")
    print(f"Valid molecules: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")
    
    if 'scaffold_retained' in df.columns:
        scaffold_count = df['scaffold_retained'].sum()
        print(f"Scaffold retained: {scaffold_count}/{valid_count} ({100*scaffold_count/max(valid_count,1):.1f}%)")
    
    if properties:
        for prop in properties.keys():
            if f'{prop}_satisfied' in df.columns:
                satisfied = df[f'{prop}_satisfied'].sum()
                print(f"{prop} satisfied: {satisfied}/{valid_count} ({100*satisfied/max(valid_count,1):.1f}%)")
    
    # Visualize molecules
    if args.visualize:
        valid_mols = []
        for smiles in df[df['valid']]['canonical_smiles'][:12]:  # First 12 valid
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_mols.append(mol)
        
        if valid_mols:
            img = Draw.MolsToGridImage(valid_mols, molsPerRow=3, subImgSize=(300, 300))
            img_path = args.output.replace('.csv', '_grid.png')
            img.save(img_path)
            logger.info(f"Molecular grid saved to {img_path}")


if __name__ == "__main__":
    main()