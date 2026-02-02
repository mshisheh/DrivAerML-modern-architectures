# Train.py Compatibility Updates

## Overview
Updated all data loaders to ensure backward compatibility with existing train.py files. The train.py files expect certain function names and constants that differ from the adapted data loaders.

## Changes Made

### 1. ✅ GraphCast_SurfaceFields/data_loader.py
**Issue:** train.py imports `load_design_ids`, but data_loader has `load_run_ids`

**Solution:** Added backward compatibility alias
```python
# Backward compatibility alias for train.py
load_design_ids = load_run_ids
```

**Exports:**
- `create_dataloaders()` - Main function
- `load_run_ids()` - Loads train/val/test IDs from files
- `load_design_ids` - Alias for `load_run_ids`

---

### 2. ✅ FIGConvNet_SurfaceFields/data_loader.py
**Issue:** Had `load_design_ids()` reading from wrong files (design_ids vs run_ids) and returning strings instead of ints

**Solution:** Replaced function to use correct files and return integers
```python
def load_run_ids(split_dir: str) -> Tuple[List[int], List[int], List[int]]:
    """Load train/val/test run IDs from text files for DrivAerML."""
    def read_ids(filename):
        with open(os.path.join(split_dir, filename), 'r') as f:
            return [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    
    train_ids = read_ids('train_run_ids.txt')
    val_ids = read_ids('val_run_ids.txt')
    test_ids = read_ids('test_run_ids.txt')
    return train_ids, val_ids, test_ids

# Backward compatibility alias for train.py
load_design_ids = load_run_ids
```

**Exports:**
- `create_dataloaders()` - Main function
- `load_run_ids()` - Loads train/val/test IDs
- `load_design_ids` - Alias for `load_run_ids`

---

### 3. ✅ Transolver_SurfaceFields/data_loader.py
**Issue:** None - already exports required functions

**Exports:**
- `TransolverDataset` - Dataset class
- `collate_fn` - Custom collation function
- `create_dataloaders()` - Main function

**train.py imports:**
```python
from data_loader import TransolverDataset, collate_fn
from data_loader import create_dataloaders  # Inside function
```

---

### 4. ✅ RegDGCNN_SurfaceFields/data_loader.py
**Issue:** train.py imports `get_dataloaders`, `PRESSURE_MEAN`, `PRESSURE_STD`, but data_loader only has `create_dataloaders`

**Solution:** Added wrapper function and module-level constants
```python
# Backward compatibility: Export normalization constants
PRESSURE_MEAN = None
PRESSURE_STD = None

def get_dataloaders(...) -> Tuple:
    """Wrapper for create_dataloaders() to match train.py API."""
    global PRESSURE_MEAN, PRESSURE_STD
    
    train_loader, val_loader, test_loader = create_dataloaders(...)
    
    # Export normalization constants from training dataset
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    
    return train_loader, val_loader, test_loader
```

**Exports:**
- `create_dataloaders()` - Original function
- `get_dataloaders()` - Wrapper for train.py
- `PRESSURE_MEAN` - Updated after calling get_dataloaders()
- `PRESSURE_STD` - Updated after calling get_dataloaders()

---

### 5. ✅ NeuralOperator_SurfaceFields/data_loader.py
**Issue:** Same as RegDGCNN - train.py imports `get_dataloaders`, `PRESSURE_MEAN`, `PRESSURE_STD`

**Solution:** Same wrapper pattern as RegDGCNN
```python
# Backward compatibility: Export normalization constants
PRESSURE_MEAN = None
PRESSURE_STD = None

def get_dataloaders(...) -> Tuple:
    """Wrapper for create_dataloaders() to match train.py API."""
    global PRESSURE_MEAN, PRESSURE_STD
    
    train_loader, val_loader, test_loader = create_dataloaders(...)
    
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    
    return train_loader, val_loader, test_loader
```

**Exports:**
- `create_dataloaders()` - Original function
- `get_dataloaders()` - Wrapper for train.py
- `PRESSURE_MEAN` - Updated after calling get_dataloaders()
- `PRESSURE_STD` - Updated after calling get_dataloaders()

---

### 6. ✅ ABUPT_SurfaceFields/data_loader.py
**Issue:** train.py imports `get_dataloaders`, `PRESSURE_MEAN`, `PRESSURE_STD`, `SurfaceFieldDataset`, `create_subset`

**Solution:** Added wrapper function, constants, and new `create_subset()` function
```python
# Backward compatibility: Export normalization constants
PRESSURE_MEAN = None
PRESSURE_STD = None

def get_dataloaders(...) -> Tuple:
    """Wrapper for create_dataloaders() to match train.py API."""
    global PRESSURE_MEAN, PRESSURE_STD
    
    train_loader, val_loader, test_loader = create_dataloaders(...)
    
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    
    return train_loader, val_loader, test_loader

def create_subset(dataset: SurfaceFieldDataset, run_ids: List[int]) -> SurfaceFieldDataset:
    """Create a subset of the dataset with specific run IDs."""
    subset = SurfaceFieldDataset(
        data_dir=dataset.data_dir,
        run_ids=run_ids,
        num_points=dataset.num_points,
        load_wss=dataset.load_wss,
        normalize=False,
    )
    
    # Share normalization stats if available
    if dataset.pressure_mean is not None:
        subset.pressure_mean = dataset.pressure_mean
        subset.pressure_std = dataset.pressure_std
        # ... etc
    
    return subset
```

**Exports:**
- `SurfaceFieldDataset` - Dataset class (already existed)
- `create_dataloaders()` - Original function
- `get_dataloaders()` - Wrapper for train.py
- `create_subset()` - NEW - for creating dataset subsets
- `PRESSURE_MEAN` - Updated after calling get_dataloaders()
- `PRESSURE_STD` - Updated after calling get_dataloaders()

---

### 7. ⚠️ Transolver++_SurfaceFields/data_loader.py
**Status:** No train.py file found yet

**Current Exports:**
- `TransolverPlusPlusDataset` - Dataset class
- `collate_fn` - Custom collation function
- `create_dataloaders()` - Main function

---

### 8. ⚠️ MeshGraphNet_SurfaceFields/data_loader.py
**Status:** No train.py file found yet

**Current Exports:**
- `MeshGraphDataset` - Dataset class
- `create_dataloaders()` - Main function

---

## Summary

### ✅ Complete - Ready to Use (6/8 models)
1. GraphCast - Added `load_design_ids` alias
2. FIGConvNet - Updated `load_run_ids` and added alias
3. Transolver - Already compatible
4. RegDGCNN - Added `get_dataloaders` wrapper and constants
5. NeuralOperator - Added `get_dataloaders` wrapper and constants
6. ABUPT - Added `get_dataloaders`, constants, and `create_subset()`

### ⚠️ Missing train.py (2/8 models)
7. Transolver++ - Data loader ready, need train.py
8. MeshGraphNet - Data loader ready, need train.py

## Key Patterns

### Pattern 1: Function Name Compatibility
When train.py expects a different function name, add an alias:
```python
# Backward compatibility alias
load_design_ids = load_run_ids
```

### Pattern 2: Module-Level Constants
When train.py imports constants, export them from a wrapper:
```python
PRESSURE_MEAN = None
PRESSURE_STD = None

def get_dataloaders(...):
    global PRESSURE_MEAN, PRESSURE_STD
    train_loader, val_loader, test_loader = create_dataloaders(...)
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    return train_loader, val_loader, test_loader
```

### Pattern 3: Wrapper Functions
Keep original function, add wrapper for compatibility:
```python
def create_dataloaders(...):  # Original
    # Implementation
    return train_loader, val_loader, test_loader

def get_dataloaders(...):  # Wrapper for backward compatibility
    global CONSTANTS
    loaders = create_dataloaders(...)
    CONSTANTS = loaders[0].dataset.stats
    return loaders
```

## Testing Checklist

Before running train.py for each model, verify:
- [ ] All imports in train.py match exports in data_loader.py
- [ ] Function signatures are compatible
- [ ] Constants are properly exported
- [ ] Data files exist in expected locations
- [ ] Dependencies are installed (torch, torch-geometric, pyvista, etc.)

## Next Steps

1. Test GraphCast training script
2. Test FIGConvNet training script
3. Test Transolver training script
4. Test RegDGCNN training script
5. Test NeuralOperator training script
6. Test ABUPT training script
7. Create train.py for Transolver++
8. Create train.py for MeshGraphNet
