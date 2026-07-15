import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import glob
import matplotlib.gridspec as gridspec
import zipfile
import os
from pathlib import Path

# --- Custom Function: Destriping & Flattening ---
def clean_and_flatten_grid(grid_data):
    """
    1. Calculates the background (median) for each horizontal row (Declination).
    2. Aligns all rows to the same global background level.
    This removes horizontal stripes AND flattens the 'Zenith Brightness' bump.
    """
    print("   -> Running Destriping algorithm...")
    
    # Calculate the median value of each row (ignoring NaNs)
    # axis=1 means we look across Right Ascension for each Declination strip
    row_backgrounds = np.nanmedian(grid_data, axis=1)
    
    # Calculate the global median of the whole map
    global_background = np.nanmedian(grid_data)
    
    # Calculate the offset needed for each row
    # reshapes to (500, 1) so we can subtract from the 2D grid
    offsets = (row_backgrounds - global_background)[:, np.newaxis]
    
    # Apply correction: Subtract the offset to level the data
    cleaned_grid = grid_data - offsets
    
    return cleaned_grid

# --- 1. Load and Stitch Data ---
print("Scanning for summary files from zip archive...")

zip_file = "gbd_horn_All_drift_scan_data_summary.zip"
extract_dir = "drift_scan_data"

# Extract zip if it exists
if os.path.exists(zip_file):
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}/")
    all_files = glob.glob(os.path.join(extract_dir, "**/*.csv"), recursive=True)
else:
    print(f"ZIP file {zip_file} not found. Searching for local CSV files...")
    all_files = glob.glob("*summary*.csv")

print(f"Found {len(all_files)} files.")

df_list = []
required_cols = ['Right Ascension (RA)', 'Declination (DEC)', 'Signal Power']

for filename in all_files:
    try:
        temp_df = pd.read_csv(filename)
        if set(required_cols).issubset(temp_df.columns):
            temp_df = temp_df[required_cols]
            for col in required_cols:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            temp_df = temp_df.dropna()
            df_list.append(temp_df)
            print(f"Loaded: {filename} ({len(temp_df)} pts)")
        else:
            print(f"SKIPPED: {filename}")
    except Exception as e:
        print(f"ERROR {filename}: {e}")

if not df_list:
    print("CRITICAL ERROR: No data.")
    exit()

full_data = pd.concat(df_list, ignore_index=True)

# --- 2. Map Boundaries ---
min_ra, max_ra = 0, 360
min_dec = full_data['Declination (DEC)'].min() - 5
max_dec = full_data['Declination (DEC)'].max() + 5

# --- 3. Create Grid ---
ra_grid = np.linspace(min_ra, max_ra, 1000) 
dec_grid = np.linspace(min_dec, max_dec, 500) 
RA_grid, DEC_grid = np.meshgrid(ra_grid, dec_grid)

# --- 4. Interpolate ---
print("Gridding data...")
grid_intensity = griddata(
    (full_data['Right Ascension (RA)'], full_data['Declination (DEC)']), 
    full_data['Signal Power'], 
    (RA_grid, DEC_grid), 
    method='linear' 
)

# --- NEW STEP: Apply Cleaning ---
print("Applying Statistical Destriping...")
# We assume grid_intensity has Dec as rows (axis 0) and RA as cols (axis 1)
# Because of how meshgrid works with griddata, we verify shape matches dec_grid
if grid_intensity.shape[0] != len(dec_grid):
    # If transposed, flip it for the function, then flip back
    grid_intensity = clean_and_flatten_grid(grid_intensity.T).T
else:
    grid_intensity = clean_and_flatten_grid(grid_intensity)

# --- 5. Smoothing ---
sigma = 4 # Slightly reduced sigma to see sharper details after cleaning
print(f"Applying Gaussian Filter (Sigma={sigma})...")
smoothed_map = gaussian_filter(grid_intensity, sigma=sigma)

# --- 6. Plotting ---
print("Generating Plot...")
fig = plt.figure(figsize=(15, 6))

gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0.3], hspace=0.3)

ax = fig.add_subplot(gs[0])

cf = ax.pcolormesh(RA_grid, DEC_grid, smoothed_map, shading='auto', cmap='turbo')

ax.set_title("Sky Map ", fontsize=26)
ax.set_xlabel("Right Ascension (degrees)", fontsize=22)
ax.set_ylabel("Declination (degrees)", fontsize=22)
ax.set_xlim(0, 360)
ax.set_ylim(min_dec, max_dec)
ax.tick_params(axis='both', which='major', labelsize=16)

cax = fig.add_subplot(gs[1])

cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
cbar.set_label("Signal Strength", fontsize=20)

plt.show()
print("Done!")