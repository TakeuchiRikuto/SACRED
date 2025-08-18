#!/usr/bin/env python3
"""Test RDKit installation and attributes"""

import sys

print("Testing RDKit...")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    print("✓ RDKit imported successfully")
    
    # Test molecule
    mol = Chem.MolFromSmiles("CCO")
    
    # Test basic descriptors
    print(f"✓ MolWt: {Descriptors.MolWt(mol):.2f}")
    print(f"✓ LogP: {Descriptors.MolLogP(mol):.2f}")
    
    # Test FractionCSP3 (different versions have different names)
    if hasattr(Descriptors, 'FractionCSP3'):
        print(f"✓ FractionCSP3: {Descriptors.FractionCSP3(mol):.2f}")
    elif hasattr(Descriptors, 'FractionCsp3'):
        print(f"✓ FractionCsp3: {Descriptors.FractionCsp3(mol):.2f}")
    else:
        print("⚠ FractionCSP3 not available in this RDKit version")
    
    # Check available descriptors
    print(f"\nRDKit version info:")
    import rdkit
    print(f"  Version: {rdkit.__version__}")
    
    # List descriptors containing 'Fraction'
    fraction_descriptors = [d for d in dir(Descriptors) if 'Fraction' in d]
    print(f"\nAvailable Fraction descriptors:")
    for desc in fraction_descriptors:
        print(f"  - {desc}")
    
except ImportError as e:
    print(f"✗ RDKit import error: {e}")
    print("\nPlease install RDKit:")
    print("  conda install -c conda-forge rdkit")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!")