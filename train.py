#!/usr/bin/env python3
"""
Training script for SACRED model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging
from tqdm import tqdm
import json

from model.sacred_model import SACRED, SACREDLoss
from model.data_processing_simple import (
    SimpleMolecularFeaturizer as MolecularFeaturizer,
    SimpleADMETCalculator as ADMETCalculator,
    SimpleScaffoldExtractor as ScaffoldExtractor,
    SimpleDataCollator as DataCollator,
    SimpleSMILESTokenizer as SMILESTokenizer
)
from evaluation.metrics import ConstraintEvaluator, PerformanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularDataset(Dataset):
    """Dataset for molecular generation"""
    
    def __init__(self, data_file: str, featurizer: MolecularFeaturizer, 
                 admet_calc: ADMETCalculator):
        self.featurizer = featurizer
        self.admet_calc = admet_calc
        
        # Load data
        with open(data_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(self.data)} molecules from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract scaffold
        scaffold = ScaffoldExtractor.extract_scaffold(item['smiles'])
        if scaffold is None:
            scaffold = 'c1ccccc1'  # Default scaffold
        
        # Calculate properties
        properties = self.admet_calc.calculate_properties(item['smiles'])
        if properties is None:
            properties = np.ones(13) * 0.5  # Default properties
        
        return {
            'smiles': item['smiles'],
            'scaffold': scaffold,
            'properties': properties,
            'target_smiles': item['smiles']  # For teacher forcing
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        scaffold_graphs = batch['scaffold_graphs'].to(device)
        properties = batch['properties'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        
        # Forward pass
        outputs = model(
            smiles=batch['smiles'],
            scaffold_graphs=scaffold_graphs,
            properties=properties,
            target_smiles=target_tokens
        )
        
        # Calculate loss
        losses = criterion(
            outputs,
            {'fused_input': outputs['fused_representation'], 
             'target_smiles': target_tokens}
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, evaluator, device):
    """Evaluate model performance"""
    model.eval()
    
    all_generated = []
    all_scaffolds = []
    all_properties = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            scaffold_graphs = batch['scaffold_graphs'].to(device)
            properties = batch['properties'].to(device)
            
            # Generate molecules
            generated = model.generate(
                scaffold_graphs=scaffold_graphs,
                properties=properties,
                num_samples=5,
                temperature=0.8
            )
            
            # Store for evaluation
            tokenizer = SMILESTokenizer()
            for gen_batch in generated:
                for tokens in gen_batch:
                    smiles = tokenizer.decode(tokens.cpu().numpy())
                    all_generated.append(smiles)
            
            all_scaffolds.extend(batch['scaffold'])
            all_properties.extend(batch['properties'].cpu().numpy())
    
    # Calculate metrics
    metrics = evaluator.evaluate_batch(
        all_generated,
        target_scaffolds=all_scaffolds
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SACRED model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--train_data', type=str, required=True, help='Training data file')
    parser.add_argument('--val_data', type=str, required=True, help='Validation data file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SACRED(config)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data processing components
    featurizer = MolecularFeaturizer()
    admet_calc = ADMETCalculator()
    tokenizer = SMILESTokenizer()
    collator = DataCollator()  # SimpleDataCollator takes no arguments
    
    # Create datasets
    train_dataset = MolecularDataset(args.train_data, featurizer, admet_calc)
    val_dataset = MolecularDataset(args.val_data, featurizer, admet_calc)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.collate_batch,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator.collate_batch,
        num_workers=4
    )
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = SACREDLoss()
    evaluator = ConstraintEvaluator()
    tracker = PerformanceTracker()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    best_score = 0
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, val_loader, evaluator, device)
            logger.info(f"Validation metrics: {metrics}")
            
            # Track performance
            tracker.update(epoch, metrics)
            
            # Save best model
            score = metrics.get('validity', 0) * 0.3 + \
                   metrics.get('uniqueness', 0) * 0.2 + \
                   metrics.get('diversity', 0) * 0.2 + \
                   metrics.get('scaffold_retention', 0) * 0.3
            
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
                logger.info(f"Saved best model with score: {score:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Plot training history
    tracker.plot_history(str(output_dir / 'training_history.png'))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()