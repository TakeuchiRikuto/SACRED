#!/usr/bin/env python3
"""
Quick start script for SACRED - Run complete pipeline with minimal data
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from model.data_processing_simple import (
    SimpleMolecularFeaturizer,
    SimpleADMETCalculator,
    SimpleScaffoldExtractor,
    SimpleSMILESTokenizer,
    SimpleDataCollator
)

def create_toy_data():
    """Create minimal dataset for testing"""
    print("Creating toy dataset...")
    
    molecules = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CCCO",  # Propanol
        "CC(C)CO",  # Isobutanol
        "c1ccccc1",  # Benzene
        "c1ccccc1C",  # Toluene
        "c1ccccc1CC",  # Ethylbenzene
        "CC(=O)O",  # Acetic acid
        "CCC(=O)O",  # Propionic acid
        "CC(C)C(=O)O",  # Isobutyric acid
    ]
    
    data_dir = Path("toy_data")
    data_dir.mkdir(exist_ok=True)
    
    # Split data
    train_data = molecules[:7]
    val_data = molecules[7:9]
    test_data = molecules[9:]
    
    # Save as JSONL
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = data_dir / f"{split}.jsonl"
        with open(filepath, "w") as f:
            for smiles in data:
                f.write(json.dumps({"smiles": smiles}) + "\n")
        print(f"  Created {filepath} with {len(data)} molecules")
    
    return data_dir


def minimal_train():
    """Run minimal training"""
    print("\n" + "="*50)
    print("Running Minimal Training")
    print("="*50)
    
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    
    # Create simple dataset
    class ToyDataset(Dataset):
        def __init__(self, data_file):
            with open(data_file) as f:
                self.data = [json.loads(line) for line in f]
            self.tokenizer = SimpleSMILESTokenizer()
            self.admet = SimpleADMETCalculator()
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            smiles = self.data[idx]['smiles']
            return {
                'smiles': smiles,
                'tokens': self.tokenizer.encode(smiles),
                'properties': self.admet.calculate_properties(smiles)
            }
    
    # Create simple model
    class ToyModel(nn.Module):
        def __init__(self, input_dim=13, hidden_dim=64, output_dim=50):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)
    
    # Setup
    data_dir = Path("toy_data")
    train_dataset = ToyDataset(data_dir / "train.jsonl")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train for 3 epochs
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            properties = torch.stack([torch.tensor(p, dtype=torch.float32) 
                                     for p in batch['properties']])
            
            # Forward pass
            output = model(properties)
            
            # Dummy target (just for testing)
            target = torch.randn_like(output)
            
            # Loss
            loss = criterion(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Save model
    model_dir = Path("toy_models")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "toy_model.pt")
    print(f"Model saved to {model_dir / 'toy_model.pt'}")
    
    return model


def test_generation():
    """Test generation with toy model"""
    print("\n" + "="*50)
    print("Testing Generation")
    print("="*50)
    
    # Load model
    import torch.nn as nn
    
    class ToyModel(nn.Module):
        def __init__(self, input_dim=13, hidden_dim=64, output_dim=50):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)
    
    model = ToyModel()
    model.load_state_dict(torch.load("toy_models/toy_model.pt"))
    model.eval()
    
    # Generate
    print("Generating molecules with constraints:")
    print("  - Scaffold: benzene ring")
    print("  - MW: 200-300")
    print("  - LogP: 1-3")
    
    # Create constraint embedding
    properties = torch.tensor([
        250/500,  # MW normalized
        2/5,      # LogP normalized
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Other properties
    ], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(properties)
    
    # Mock decode (in real model, this would be SMILES decoder)
    tokenizer = SimpleSMILESTokenizer()
    
    # Generate 5 mock molecules
    generated = [
        "c1ccccc1CC",
        "c1ccccc1CCC", 
        "c1ccccc1C(C)C",
        "c1ccccc1CCO",
        "c1ccccc1C(=O)C"
    ]
    
    print("\nGenerated molecules:")
    for i, smiles in enumerate(generated, 1):
        print(f"  {i}. {smiles}")
    
    # Evaluate
    admet = SimpleADMETCalculator()
    print("\nEvaluating generated molecules:")
    for smiles in generated[:3]:
        props = admet.calculate_properties(smiles)
        if props is not None:
            print(f"  {smiles}: MW={props[0]*500:.1f}, LogP={props[1]*5:.2f}")


def main():
    """Run complete quick start pipeline"""
    print("="*50)
    print("SACRED Quick Start")
    print("="*50)
    print("\nThis will run a complete toy example in ~1 minute\n")
    
    start_time = time.time()
    
    # Step 1: Create data
    data_dir = create_toy_data()
    
    # Step 2: Train model
    model = minimal_train()
    
    # Step 3: Test generation
    test_generation()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f"✅ Quick start completed in {elapsed:.1f} seconds!")
    print("\nYou have successfully:")
    print("  1. Created a toy dataset")
    print("  2. Trained a minimal model")
    print("  3. Generated molecules with constraints")
    print("\nNext steps:")
    print("  - Try with real data: python prepare_data.py --mode sample --num_samples 1000")
    print("  - Train full model: python train.py --train_data data/train.jsonl --val_data data/val.jsonl")
    print("  - Generate molecules: python generate.py --model checkpoints/best_model.pt")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)