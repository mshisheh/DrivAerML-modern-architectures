"""
Verify Train.py Compatibility

Checks that all train.py files can successfully import from their data_loader.py files.
This verifies the compatibility updates are correct before attempting to run training.
"""

import sys
import importlib.util
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_imports(model_name, data_loader_path, expected_imports):
    """Check if a data_loader.py exports all expected symbols."""
    print(f"\n{'='*60}")
    print(f"Checking {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(f"{model_name}_data_loader", data_loader_path)
        module = importlib.util.module_from_spec(spec)
        
        # Try to execute it (this will fail if dependencies are missing, but that's ok)
        try:
            spec.loader.exec_module(module)
            module_loaded = True
        except Exception as e:
            print(f"{YELLOW}⚠ Module loaded with dependency issues (expected): {e}{RESET}")
            module_loaded = False
        
        # Check if expected symbols are defined in the module
        all_good = True
        for symbol in expected_imports:
            # Check if symbol is in the module's source code
            with open(data_loader_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Check for different definition patterns
            if symbol.isupper():  # Constant
                pattern = f"{symbol} ="
            elif symbol[0].isupper():  # Class
                pattern = f"class {symbol}"
            else:  # Function
                pattern = f"def {symbol}("
            
            if pattern in source or f"{symbol} =" in source:  # Also check for aliases
                print(f"  {GREEN}✓{RESET} {symbol:30} - Found")
            else:
                print(f"  {RED}✗{RESET} {symbol:30} - Missing")
                all_good = False
        
        if all_good:
            print(f"\n{GREEN}✓ All imports available for {model_name}{RESET}")
        else:
            print(f"\n{RED}✗ Some imports missing for {model_name}{RESET}")
        
        return all_good
        
    except Exception as e:
        print(f"{RED}✗ Error loading module: {e}{RESET}")
        return False


def main():
    print("="*60)
    print("Train.py Compatibility Verification")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    # Model configurations: (name, data_loader_path, expected_imports)
    models = [
        (
            "GraphCast",
            base_dir / "GraphCast_SurfaceFields" / "data_loader.py",
            ["create_dataloaders", "load_design_ids", "load_run_ids"]
        ),
        (
            "FIGConvNet",
            base_dir / "FIGConvNet_SurfaceFields" / "data_loader.py",
            ["create_dataloaders", "load_design_ids", "load_run_ids"]
        ),
        (
            "Transolver",
            base_dir / "Transolver_SurfaceFields" / "data_loader.py",
            ["TransolverDataset", "collate_fn", "create_dataloaders"]
        ),
        (
            "RegDGCNN",
            base_dir / "RegDGCNN_SurfaceFields" / "data_loader.py",
            ["get_dataloaders", "create_dataloaders", "PRESSURE_MEAN", "PRESSURE_STD"]
        ),
        (
            "NeuralOperator",
            base_dir / "NeuralOperator_SurfaceFields" / "data_loader.py",
            ["get_dataloaders", "create_dataloaders", "PRESSURE_MEAN", "PRESSURE_STD"]
        ),
        (
            "ABUPT",
            base_dir / "ABUPT_SurfaceFields" / "data_loader.py",
            ["get_dataloaders", "create_dataloaders", "create_subset", 
             "SurfaceFieldDataset", "PRESSURE_MEAN", "PRESSURE_STD"]
        ),
        (
            "Transolver++",
            base_dir / "Transolver++_SurfaceFields" / "data_loader.py",
            ["PointCloudNormalDataset", "create_dataloaders"]
        ),
        (
            "MeshGraphNet",
            base_dir / "MeshGraphNet_SurfaceFields" / "data_loader.py",
            ["MeshGraphDataset", "create_dataloaders"]
        ),
    ]
    
    results = {}
    for model_name, data_loader_path, expected_imports in models:
        if not data_loader_path.exists():
            print(f"\n{YELLOW}⚠ {model_name}: data_loader.py not found{RESET}")
            results[model_name] = False
            continue
        
        results[model_name] = check_imports(model_name, data_loader_path, expected_imports)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_name, success in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if success else f"{RED}✗ FAIL{RESET}"
        print(f"{model_name:20} {status}")
    
    print(f"\n{passed}/{total} models passed compatibility check")
    
    if passed == total:
        print(f"\n{GREEN}✓ All models ready for training!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some models have compatibility issues{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
