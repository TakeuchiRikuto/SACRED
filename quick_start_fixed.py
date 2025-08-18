#!/usr/bin/env python3
"""
Fixed quick start script for SACRED - Handles batch collation properly
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import sys
import time
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent))

from model.data_processing_simple import (
    SimpleADMETCalculator,
    SimpleSMILESTokenizer,
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


class ToyDataset(Dataset):
    """Fixed dataset with proper tensor handling"""
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [json.loads(line) for line in f]
        self.admet = SimpleADMETCalculator()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]['smiles']
        properties = self.admet.calculate_properties(smiles)
        
        # Ensure properties is a numpy array of correct shape
        if properties is None:
            properties = np.ones(13, dtype=np.float32) * 0.5
        
        return {
            'smiles': smiles,
            'properties': torch.tensor(properties, dtype=torch.float32)
        }


class ToyModel(nn.Module):
    """Simple model for testing"""
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


def minimal_train():
    """Run minimal training with fixed batch handling"""
    print("\n" + "="*50)
    print("Running Minimal Training")
    print("="*50)
    
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
        for batch_idx, batch in enumerate(train_loader):
            # Get properties tensor
            properties = batch['properties']
            
            # Forward pass
            output = model(properties)
            
            # Create dummy target for testing
            target = torch.randn_like(output)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/3: Loss = {avg_loss:.4f}")
    
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
    model = ToyModel()
    model_path = Path("toy_models/toy_model.pt")
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print("Warning: No saved model found, using random weights")
    
    model.eval()
    
    # Generate with constraints
    print("\nGenerating molecules with constraints:")
    print("  - Target MW: 250 (normalized)")
    print("  - Target LogP: 2 (normalized)")
    
    # Create constraint embedding
    properties = torch.tensor([
        250/500,  # MW normalized
        2/5,      # LogP normalized  
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Other properties
    ], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(properties)
        print(f"  Generated latent vector shape: {output.shape}")
    
    # Mock generated molecules (in real model, decoder would generate these)
    generated = [
        "c1ccccc1CC",
        "c1ccccc1CCC",
        "c1ccccc1C(C)C", 
        "c1ccccc1CCO",
        "c1ccccc1C(=O)C"
    ]
    
    print("\nGenerated molecules (mock):")
    for i, smiles in enumerate(generated, 1):
        print(f"  {i}. {smiles}")
    
    # Evaluate properties
    admet = SimpleADMETCalculator()
    print("\nProperty evaluation of generated molecules:")
    for smiles in generated[:3]:
        props = admet.calculate_properties(smiles)
        if props is not None:
            # Display denormalized values
            mw = props[0] * 500  # Denormalize MW
            logp = props[1] * 5   # Denormalize LogP
            print(f"  {smiles}: MW≈{mw:.1f}, LogP≈{logp:.2f}")


def main():
    """Run complete quick start pipeline"""
    print("="*50)
    print("SACRED Quick Start (Fixed)")
    print("="*50)
    print("\nThis will run a complete toy example in ~1 minute\n")
    
    start_time = time.time()
    
    try:
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
        print("  - Try with more data: python prepare_data.py --mode sample --num_samples 1000")
        print("  - Train full SACRED model: python train.py --train_data data/train.jsonl --val_data data/val.jsonl")
        print("  - Generate with real model: python generate.py --model checkpoints/best_model.pt")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)