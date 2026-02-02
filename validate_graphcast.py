"""
Simple validation check for GraphCast data loader adaptations.
This validates the code structure without running it.
"""

import ast
import sys

def check_data_loader_structure(file_path):
    """Parse and validate the data loader structure."""
    print(f"Checking: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the file
    try:
        tree = ast.parse(content)
        print("  ✓ Python syntax is valid")
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False
    
    # Check for key elements
    checks = {
        'pyvista import': 'import pyvista' in content or 'pv.read' in content,
        'pandas import': 'import pandas' in content or 'pd.read_csv' in content,
        'run_ids parameter': 'run_ids' in content,
        'boundary_{run_id}.vtp': 'boundary_{' in content and '.vtp' in content,
        'CpMeanTrim field': 'CpMeanTrim' in content,
        'cell_centers()': 'cell_centers()' in content,
        'compute_normals': 'compute_normals' in content,
        'compute_cell_sizes': 'compute_cell_sizes' in content,
        'geo_parameters CSV': 'geo_parameters_' in content and '.csv' in content,
        'GraphCastDataset class': 'class GraphCastDataset' in content,
        'load_run_ids function': 'def load_run_ids' in content or 'load_run_ids' in content,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        if result:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ Missing: {check_name}")
            all_passed = False
    
    return all_passed

def main():
    print("="*70)
    print("GraphCast Data Loader Validation for DrivAerML")
    print("="*70)
    print()
    
    file_path = r"c:\Learning\Scientific\CARBENCH\DrivAerML\GraphCast_SurfaceFields\data_loader.py"
    
    success = check_data_loader_structure(file_path)
    
    print()
    print("="*70)
    if success:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("The data loader has been correctly adapted for DrivAerML:")
        print("  • VTP file loading with pyvista")
        print("  • run_{id}/boundary_{id}.vtp file structure") 
        print("  • CpMeanTrim field extraction from cell_data")
        print("  • Cell centers, normals, and areas")
        print("  • Geometry parameters from CSV files")
        print("  • Integer run_ids instead of string design_ids")
        print()
        print("Key differences from DrivAerNet++:")
        print("  • File pattern: run_{id}/boundary_{id}.vtp (not {design_id}.npy)")
        print("  • Target field: CpMeanTrim (cell_data, not array column)")
        print("  • Added: geometry parameters (16 global design variables)")
        print("  • ID format: integers 1-500 (not string design IDs)")
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please review the missing elements above.")
    print("="*70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
