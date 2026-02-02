# Download DrivAerNet++ Surface Data Only

For surface pressure prediction, you only need the **surface mesh** and **pressure field** files, not the 406 GB volume data.

## What You Need (Much Smaller!)

| Data Type | Size | Priority | Files |
|-----------|------|----------|-------|
| **Surface Pressure** | ~20-30 GB | ✅ **REQUIRED** | `.vtp` or `.vtk` files |
| **Surface Meshes (STL)** | ~5-10 GB | ✅ **REQUIRED** | `.stl` files |
| **Coefficients (CSV)** | <1 MB | ✅ **REQUIRED** | `.csv` files |
| Volume Data | 406 GB | ❌ **NOT NEEDED** | `.vtu` files |

**Total needed: ~30-40 GB** instead of 406 GB!

## Download Instructions

### Step 1: Go to Harvard Dataverse

**URL**: https://dataverse.harvard.edu/dataverse/DrivAerNet

### Step 2: Look for These Specific Files

On the dataverse page, look for archives named something like:

- `Surface_Pressure.zip` or `PressureVTK.zip` (~20-30 GB)
- `Surface_Meshes.zip` or `STL_Combined.zip` (~5-10 GB)
- `Coefficients.csv` (<1 MB)

**Note**: The exact filenames may vary. Look for "surface" or "boundary" in the name.

### Step 3: Download via Browser or wget

#### Option A: Browser Download (Simple)
1. Click on each file
2. Click "Download"
3. Wait for download to complete

#### Option B: wget/curl (Faster, resumable)

Once you have the direct URLs from Harvard Dataverse, use:

```powershell
# Install wget for Windows if needed
# winget install GnuWin32.Wget

# Download surface pressure data (replace URL with actual link from dataverse)
wget -c "https://dataverse.harvard.edu/api/access/datafile/XXXXX" -O Surface_Pressure.zip

# Download surface meshes
wget -c "https://dataverse.harvard.edu/api/access/datafile/YYYYY" -O Surface_Meshes.zip
```

The `-c` flag allows resume if download is interrupted.

### Step 4: Extract to Your Data Directory

```powershell
# Create data directory
cd C:\Learning\Scientific\CARBENCH\DrivAerNet
New-Item -ItemType Directory -Force -Path "data\PressureVTK"
New-Item -ItemType Directory -Force -Path "data\STL"

# Extract (assuming you have 7zip or similar)
# Adjust paths based on where you downloaded
7z x Surface_Pressure.zip -o"data\PressureVTK"
7z x Surface_Meshes.zip -o"data\STL"
```

### Step 5: Verify Data Structure

After extraction, you should have:

```
C:\Learning\Scientific\CARBENCH\DrivAerNet\
├── data\
│   ├── PressureVTK\
│   │   ├── DrivAer_F_D_WM_WW_0001.vtp
│   │   ├── DrivAer_F_D_WM_WW_0002.vtp
│   │   └── ... (~8,150 files)
│   ├── STL\
│   │   ├── DrivAer_F_D_WM_WW_0001.stl
│   │   ├── DrivAer_F_D_WM_WW_0002.stl
│   │   └── ... (~8,150 files)
│   ├── DrivAerNetPlusPlus_Cd_8k.csv
│   └── DrivAerNetPlusPlus_Areas.csv
└── train_val_test_splits\
    ├── train_designs.txt
    ├── val_designs.txt
    └── test_designs.txt
```

## Alternative: Start with a Subset

If even 30-40 GB is too much, you can:

### Option 1: Download Only Validation Set (~800 designs ≈ 3-4 GB)

1. Download full surface data
2. Extract only files matching validation set:

```powershell
# Read validation design IDs
$val_ids = Get-Content "train_val_test_splits\val_designs.txt"

# Copy only validation files to a subset directory
foreach ($id in $val_ids) {
    Copy-Item "data\PressureVTK\$id.vtp" -Destination "data_subset\PressureVTK\" -ErrorAction SilentlyContinue
    Copy-Item "data\STL\$id.stl" -Destination "data_subset\STL\" -ErrorAction SilentlyContinue
}
```

### Option 2: Contact Dataset Authors

If Harvard Dataverse doesn't have surface-only downloads separated, you can:

📧 Email: **mohamed.elrefaie@mit.edu**
Subject: "DrivAerNet++ Surface-Only Data Request"

Ask if they have a surface-only subset available for download.

## What You DON'T Need

❌ **Skip these to save space:**
- Volume files (`.vtu`) - 406 GB
- Any file with "volume" in the name
- Folders named "volume" or "volumetric"
- Any 3D grid/voxel data

## Next Steps After Download

Once you have the surface data, you'll need to:

1. **Convert VTP → NPY** format (for your models)
2. **Preprocess**: Normalize coordinates, compute areas
3. **Verify**: Check against train/val/test splits

Would you like me to create a preprocessing script to convert the VTP files to the `.npy` format your models expect?
