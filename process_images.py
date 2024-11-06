import os
import cv2
import numpy as np
import shutil
from skimage import measure

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)


def rgb_to_xyz(r, g, b):
    def first(c):
        return (((c / 255.0 / 0.055) + 1) * 0.0521) ** 2.4 if c / 255.0 > 0.04045 else c / 255.0 / 12.92

    RR, GG, BB = first(r), first(g), first(b)
    X = (RR * 41.24) + (GG * 35.72) + (BB * 18.05)
    Y = (RR * 21.26) + (GG * 71.52) + (BB * 7.22)
    Z = (RR * 1.93) + (GG * 11.92) + (BB * 95.05)
    
    return round(X,2), round(Y,2), round(Z,2)

# Define the function to convert XYZ to LAB
def xyz_to_lab(X, Y, Z):
    def second(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    x, y, z = X / 95.047, Y / 100, Z / 108.883
    VarX, VarY, VarZ = second(x), second(y), second(z)

    L_star = (116 * VarY) - 16
    a_star = 500 * (VarX - VarY)
    b_star = 200 * (VarY - VarZ)
    
    return round(L_star), round(a_star), round(b_star)

def rgb_to_lab(r, g, b):
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)
    
# Function to process the uploaded image
def process_images(image, threshold_value, idx):

    #Original Image
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # Load the image in BGR format
    
    #Grayscale Image
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    #Threshold Image
    _,triangle = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7, 7), np.uint8)
    
    #Opening Image
    opening = cv2.morphologyEx(triangle, cv2.MORPH_OPEN, kernel)
    
    #Closing Image
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Extract the regions of interest (ROIs) from the original image
    extraction = cv2.bitwise_and(image_bgr, image_bgr, mask=closing)
    
    # Find contours of the ROIs
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the individual masked ROIs
    roi_rgb = []
    roi_lab = []
    roi_bi = []
    roi_diameter = []
    roi_perimeter = []
    roi_area = []

    # Iterate over contours to extract each ROI
    for i, contour in enumerate(contours):
        # Create an empty mask the same size as the original image
        mask = np.zeros_like(closing)

        # Draw the contour on the mask, filling it with white
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Bitwise AND to extract the region of interest (ROI) from the original image
        roi = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        
        # Extract only the masked region (non-zero pixels)
        roi[mask == 0] = [0, 0, 0]  # Set the background (outside contour) to black

        temp_image = os.path.join(output_dir, f'Image {idx}')
        if not os.path.exists(temp_image):
            os.makedirs(temp_image)
        cv2.imwrite(os.path.join(temp_image, f'roi_{i+1}.jpg'), roi)
            
        
        # Extract the non-zero pixel locations (where the mask is applied)
        roi_non_zero_indices = np.where(mask != 0)
        
        # Get the R, G, B values of the pixels in the ROI
        r_values = roi[roi_non_zero_indices[0], roi_non_zero_indices[1], 2]  # Red channel
        g_values = roi[roi_non_zero_indices[0], roi_non_zero_indices[1], 1]  # Green channel
        b_values = roi[roi_non_zero_indices[0], roi_non_zero_indices[1], 0]  # Blue channel

        # Calculate the average R, G, B values
        avg_r = round(np.mean(r_values))
        avg_g = round(np.mean(g_values))
        avg_b = round(np.mean(b_values))
        
        avg_l, avg_a, avg_bb = rgb_to_lab(avg_r, avg_g, avg_b)
        
        roi_rgb.append((avg_r, avg_g, avg_b))
        roi_lab.append((avg_l, avg_a, avg_bb))
        
        k = (avg_a + (1.75*avg_l))/((5.645*avg_l)+avg_a-(3.012*avg_bb))

        bi = (100*(k-0.31))/0.17
        roi_bi.append(round(bi))
        
        
    label_image = measure.label(closing)

    # Calculate properties for each labeled region
    for region in measure.regionprops(label_image):
        # Area
        area = region.area
        # Perimeter
        perimeter = region.perimeter
        # Equivalent Diameter
        diameter = region.equivalent_diameter

        roi_area.append(round(area))
        roi_perimeter.append(round(perimeter))
        roi_diameter.append(round(diameter))
        
    extraction_lab = cv2.cvtColor(extraction, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, a, and b channels
    L, a, b = cv2.split(extraction_lab)
    n = len(roi_rgb)

    # Optional: Normalize the L, a, b channels for proper visualization
    # L is already in the range 0-100 (usually), but a and b need to be adjusted from -128 to 127 to 0-255 for viewing.
    L_normalized = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_normalized = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    
    avg_r = round(np.mean([rgb[0] for rgb in roi_rgb]))
    avg_g = round(np.mean([rgb[1] for rgb in roi_rgb]))
    avg_b = round(np.mean([rgb[2] for rgb in roi_rgb]))
    avg_l = round(np.mean([lab[0] for lab in roi_lab]))
    avg_a = round(np.mean([lab[1] for lab in roi_lab]))
    avg_b = round(np.mean([lab[2] for lab in roi_lab]))
    avg_area = round(np.mean(roi_area))
    avg_perimeter = round(np.mean(roi_perimeter))
    avg_diameter = round(np.mean(roi_diameter))
    avg_bi = round(np.mean(roi_bi))
                
    temp_image = os.path.join(output_dir, f'Image {idx}')
    
    if not os.path.exists(temp_image):
        os.makedirs(temp_image)
    cv2.imwrite(os.path.join(temp_image, f'Image{idx}.jpg'), image_bgr)
    cv2.imwrite(os.path.join(temp_image, f'gray_image{idx}.jpg'), image_gray)
    cv2.imwrite(os.path.join(temp_image, f'threshold_image{idx}.jpg'), triangle)
    cv2.imwrite(os.path.join(temp_image, f'opening{idx}.jpg'), opening)
    cv2.imwrite(os.path.join(temp_image, f'closing.jpg'), closing)
    cv2.imwrite(os.path.join(temp_image, f'extraction{idx}.jpg'), extraction)
    cv2.imwrite(os.path.join(temp_image, f'L_image{idx}.jpg'), L_normalized)
    cv2.imwrite(os.path.join(temp_image, f'a_image{idx}.jpg'), a_normalized)
    cv2.imwrite(os.path.join(temp_image, f'b_image{idx}.jpg'), b_normalized)

    return avg_r, avg_g, avg_b, avg_l, avg_a, avg_b, avg_area, avg_perimeter, avg_diameter, avg_bi, n