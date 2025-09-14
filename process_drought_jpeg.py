import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- 1. Define Function to Calculate NDVI from a color image ---
def calculate_ndvi_from_jpeg(image_path):
    """
    Calculates an approximate NDVI from a standard color (RGB) image.
    This is a simulation for when separate spectral bands are not available.
    Args:
        image_path (str): Filepath to the JPEG image.
    Returns:
        numpy.ndarray: The calculated NDVI array.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Split into Red, Green, and Blue channels
    blue, green, red = cv2.split(img)

    # For RGB images, a common approximation is to use the Red channel as 'Red'
    # and a combination of other channels or a fixed value for 'NIR'.
    # A simple but effective method is to treat the Green channel's intensity as NIR.
    # Healthy vegetation reflects both green and NIR light.
    red = red.astype('float64')
    nir = green.astype('float64') # Approximating NIR with the Green channel

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    numerator = (nir - red)
    denominator = (nir + red)
    ndvi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    # Handle potential NaNs and clean up the range to [-1, 1]
    ndvi[np.isnan(ndvi)] = 0
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi

# --- 2. Define Function to Classify Drought Severity ---
def classify_drought(ndvi):
    """
    Classifies NDVI values into drought severity levels.
    Thresholds are adjusted for the JPEG approximation.
    """
    drought_map = np.zeros_like(ndvi, dtype=np.uint8)
    # These thresholds are indicative and may need tuning.
    drought_map[np.where(ndvi < 0.05)] = 4  # Severe Drought (likely non-vegetation)
    drought_map[np.where((ndvi >= 0.05) & (ndvi < 0.15))] = 3  # Moderate Drought
    drought_map[np.where((ndvi >= 0.15) & (ndvi < 0.3))] = 2  # Mild Drought
    drought_map[np.where((ndvi >= 0.3) & (ndvi < 0.4))] = 1  # Low to No Drought
    drought_map[np.where(ndvi >= 0.4)] = 0  # Healthy Vegetation
    return drought_map

# --- 3. Process a sample image from the downloaded dataset ---
# Path points to the folder containing the image classes
DATASET_DIR = r"C:\Users\ASUS\Desktop\Climate Sentinel\Satellite-Image-Classification\data"

# CORRECTED: Use 'green_area' as it exists in the dataset
VEGETATION_FOLDER = 'green_area' 

# Get a random image from the folder to analyze
try:
    folder_path = os.path.join(DATASET_DIR, VEGETATION_FOLDER)
    image_files = os.listdir(folder_path)

    if not image_files:
        raise FileNotFoundError(f"No files found in {VEGETATION_FOLDER} directory.")

    sample_image_name = random.choice(image_files)
    sample_image_path = os.path.join(folder_path, sample_image_name)
    print(f"Analyzing sample image: {sample_image_path}")

    original_image = cv2.cvtColor(cv2.imread(sample_image_path), cv2.COLOR_BGR2RGB)
    ndvi_array = calculate_ndvi_from_jpeg(sample_image_path)
    drought_severity_map = classify_drought(ndvi_array)

    # --- 4. Visualize the Output ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Satellite Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ndvi_array, cmap='RdYlGn')
    plt.colorbar(label='Approx. NDVI Value')
    plt.title('Calculated NDVI')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(drought_severity_map, cmap='YlOrRd')
    plt.colorbar(label='Drought Severity (4=Severe)')
    plt.title('Drought Severity Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("-" * 50)
    print("ERROR: Could not find the dataset directory or images.")
    print(f"Please make sure you have downloaded and unzipped the dataset,")
    print(f"and the '{DATASET_DIR}' path is correct.")
    print("-" * 50)