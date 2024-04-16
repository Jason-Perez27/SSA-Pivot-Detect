import os
import numpy as np
import rasterio
import imageio
from PIL import Image

def rescale_to_8bit(band_array, min_value=None, max_value=None):
    valid_pixels = band_array[~np.isnan(band_array)]
    if len(valid_pixels) == 0:
        min_value = 0  
        max_value = 255  
    else:
        if min_value is None:
            min_value = np.percentile(valid_pixels, 2)  
        if max_value is None:
            max_value = np.percentile(valid_pixels, 98)  

    band_array = np.clip(band_array, min_value, max_value)
    band_array -= min_value
    band_array /= (max_value - min_value)
    band_array *= 255
    band_array = np.nan_to_num(band_array).astype(np.uint8)

    return band_array

# Adjust file paths
landsat_data_dir = 'data/World_CPIS_2021/SSA_TIF_REQUEST'
output_jpeg_dir = 'data/World_CPIS_2021/Landsat_CP_Images'

# Create the output directory if it does not exist
if not os.path.exists(output_jpeg_dir):
    os.makedirs(output_jpeg_dir)

# Get a list of all TIFF files in the input directory and its subdirectories
tif_files = [os.path.join(root, filename)
             for root, _, files in os.walk(landsat_data_dir)
             for filename in files if filename.lower().endswith('.tif')]

# List of visualization names
visualization_names = ["RGB",  "NDVI", "LST"]

# Process each TIFF file
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        # Handle nodata values (if present) by setting them to NaN
        bands = [src.read(channel, masked=True).filled(np.nan) for channel in (1, 2, 3, 4, 5, 6)]
        
        # Extract the base filename
        base_filename = os.path.splitext(os.path.basename(tif_file))[0]

    for visualization_name in visualization_names:
        if visualization_name == "RGB":
            # RGB visualization: Red = R, Green = G, Blue = B
            red_rescaled = rescale_to_8bit(bands[2])
            green_rescaled = rescale_to_8bit(bands[1])
            blue_rescaled = rescale_to_8bit(bands[0])
        elif visualization_name == "NDVI":
            # NDVI calculation: (Band 4 - Band 3) / (Band 4 + Band 3)
            ndvi = (bands[3] - bands[2]) / (bands[3] + bands[2])
            # Map NDVI values to 8-bit grayscale (0-255)
            ndvi_rescaled = rescale_to_8bit(ndvi, -1, 1)
            red_rescaled = green_rescaled = blue_rescaled = ndvi_rescaled
        elif visualization_name == "LST":
            # LST visualization: Use the TIR (Band 6) directly for LST
            lst_rescaled = rescale_to_8bit(bands[5])
            # Convert LST to grayscale
            red_rescaled = green_rescaled = blue_rescaled = lst_rescaled

        # Create the output JPEG filename based on the input TIFF filename and the visualization
        output_jpeg_file = os.path.join(output_jpeg_dir, f"{base_filename}_{visualization_name}.jpg")

        # Stack the bands according to the channel assignments
        rgb_image = np.stack(
            (red_rescaled, green_rescaled, blue_rescaled),
            axis=-1
        )

        # Save the image as JPEG with better interpolation for image quality
        imageio.imsave(output_jpeg_file, rgb_image, quality=95)

# Create side-by-side images
for tif_file in tif_files:
    base_filename = os.path.splitext(os.path.basename(tif_file))[0]
    filenames_to_combine = []

    for visualization_name in visualization_names:
        filenames_to_combine.append(os.path.join(output_jpeg_dir, f"{base_filename}_{visualization_name}.jpg"))

    images = [Image.open(f) for f in filenames_to_combine]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Modify the output image size
    new_image = Image.new('RGB', (total_width * 2, max_height * 2))

    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    output_combined_file = os.path.join(output_jpeg_dir, f"{base_filename}_Combined.jpg")
    new_image.save(output_combined_file)

print("JPEG conversion completed.")
