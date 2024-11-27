import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import shutil
from pathlib import Path

# Define the function to convert RGB to XYZ color space (sRGB D65 illuminant)
def rgb_to_xyz(r, g, b):
    def first(c):
        return (
            ((((c-13.94) / (177.2-13.94) / 0.055) + 1) * 0.0521) ** 2.4
            if (c-13.94) / (177.2-13.94)> 0.04045
            else (c-13.94) / (177.2-13.94) / 12.92
        )
    def second(c):
        return (
            ((((c-11.07) / (176.52-11.07)/ 0.055) + 1) * 0.0521) ** 2.4
            if (c-11.07) / (176.52-11.07)> 0.04045
            else (c-11.07) / (176.52-11.07) / 12.92
        )
    def third(c):
        return (
            ((((c-9.1) / (175.38-9.1)/ 0.055) + 1) * 0.0521) ** 2.4
            if ((c-9.1) / (175.38-9.1)/ 0.055)> 0.04045
            else ((c-9.1) / (175.38-9.1)/ 0.055) / 12.92
        )

    RR, GG, BB = first(r), second(g), third(b)
    X = (RR * 41.24) + (GG * 35.72) + (BB * 18.05)
    Y = (RR * 21.26) + (GG * 71.52) + (BB * 7.22)
    Z = (RR * 1.93) + (GG * 11.92) + (BB * 95.05)

    return round(X, 2), round(Y, 2), round(Z, 2)


# Define the function to convert XYZ to LAB color space (sRGB D65 illuminant)
def xyz_to_lab(X, Y, Z):
    def second(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    x, y, z = X / 95.047, Y / 100, Z / 108.883
    VarX, VarY, VarZ = second(x), second(y), second(z)

    L_star = (116 * VarY) - 16
    a_star = 500 * (VarX - VarY)
    b_star = 200 * (VarY - VarZ)

    return round(L_star,2), round(a_star, 2), round(b_star,2)


# Define the function to convert RGB to LAB color space (sRGB D65 illuminant) using the above functions
def rgb_to_lab(r, g, b):
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)

def sorting_images(
    avg_r,
    avg_g,
    avg_b,
    avg_l,
    avg_a,
    avg_bb,
    avg_area,
    avg_perimeter,
    avg_diameter,
    avg_bi,
    avg_elongation,
    avg_chroma,
    avg_hue,
    avg_roundness,
    image_infos,
):

    # Combine the lists into a list of tuples for sorting based on the image number (image_info) in ascending order (image_infos) to maintain the order of the images in the final output table
    combined = list(
        zip(
            image_infos,
            avg_r,
            avg_g,
            avg_b,
            avg_l,
            avg_a,
            avg_bb,
            avg_area,
            avg_perimeter,
            avg_diameter,
            avg_bi,
            avg_elongation,
            avg_chroma,
            avg_hue,
            avg_roundness,
        )
    )

    # Sort the list of tuples by the first element (image) in ascending order and store it in a new list of tuples called sorted_combined
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Unzip the sorted_combined list of tuples into separate lists for each parameter to return them individually in the same order as the image_infos list (image number) in ascending order
    (
        image_infos,
        avg_r,
        avg_g,
        avg_b,
        avg_l,
        avg_a,
        avg_bb,
        avg_area,
        avg_perimeter,
        avg_diameter,
        avg_bi,
        avg_elongation,
        avg_chroma,
        avg_hue,
        avg_roundness,
    ) = zip(*sorted_combined)

    # Convert the lists to a list of tuples to return them as a single object in the same order as the image_infos list (image number) in ascending order
    image_infos = list(image_infos)
    avg_r = list(avg_r)
    avg_g = list(avg_g)
    avg_b = list(avg_b)
    avg_l = list(avg_l)
    avg_a = list(avg_a)
    avg_bb = list(avg_bb)
    avg_area = list(avg_area)
    avg_perimeter = list(avg_perimeter)
    avg_diameter = list(avg_diameter)
    avg_bi = list(avg_bi)
    avg_elongation = list(avg_elongation)
    avg_chroma = list(avg_chroma)
    avg_hue = list(avg_hue)
    avg_roundness = list(avg_roundness)

    return (
        avg_r,
        avg_g,
        avg_b,
        avg_l,
        avg_a,
        avg_bb,
        avg_area,
        avg_perimeter,
        avg_diameter,
        avg_bi,
        avg_elongation,
        avg_chroma,
        avg_hue,
        avg_roundness,
        image_infos,
    )

def process_image(image, threshold_value):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # Apply morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)
    return morph_close

def extract_rois_from_contours(image, processed_thresh):
    # Find contours
    contours, _ = cv2.findContours(processed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    masked_rois = []
    for contour in contours:
        # Create a bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Mask the original image
        mask = np.zeros_like(processed_thresh)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour
        masked_roi = cv2.bitwise_and(image, image, mask=mask)
        masked_rois.append(masked_roi)
        cropped_roi = masked_roi[y:y+h, x:x+w]
        rois.append(cropped_roi)
    return rois, contours, masked_rois

def add_padding_to_rois(rois, padding=10, bg_color=(14, 17, 23)):
    padded_rois = []
    for roi in rois:
        # Create a blank image with padding
        h, w, _ = roi.shape
        padded_roi = np.full((h + 2 * padding, w + 2 * padding, 3), bg_color, dtype=np.uint8)
        # Place the ROI in the center
        padded_roi[padding:padding+h, padding:padding+w] = roi
        padded_rois.append(padded_roi)
    return padded_rois

def calculate_roi_properties(roi, contour):
    # Ensure the ROI is in color, not grayscale
    if len(roi.shape) == 3 and roi.shape[2] == 3:  # Check if the ROI is color
        # Create a mask with the same size as the ROI
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)  # Mask will be black (0)
        
        # Fill the contour area with white (255)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Mask the ROI (only keep pixels inside the contour)
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        # Calculate Area
        area = np.sum(mask == 255)  # The area is the number of white pixels in the mask

        # Calculate Perimeter using the contour
        perimeter = cv2.arcLength(contour, True)

        # Calculate Equivalent Diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # Calculate Minimum Length and Maximum Length using bounding box
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        min_length = min(w, h)
        max_length = max(w, h)
        
        elongation = max_length / min_length if min_length > 0 else 0

        # Calculate RGB average of the ROI inside the contour
        avg_rgb = np.mean(masked_roi[mask == 255], axis=0).astype(int)  # Average only inside the contour
        r, g, b = avg_rgb  # Extract R, G, and B values
        
        l, a, bb = rgb_to_lab(r, g, b)
                
        k = (a + (1.75 * l)) / ((5.645 * l) + a - (0.3012 * bb))
        bi = (100 * (k - 0.31)) / 0.17
        
        chroma = np.round(np.sqrt(l**2 + a**2 + bb**2), 2)
        hue = np.round(np.degrees(np.arctan2(bb, a)), 2)
        
        roundness = (np.round(4 * np.pi * area / (perimeter**2), 2) if perimeter > 0 else 0)

        return area, perimeter, equivalent_diameter, r, g, b, l, a, bb, bi, chroma, hue, roundness, elongation
    else:
        return None, None, None, None, None, None, None, None


def clear_output_directory():
    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
def process_images(uploaded_files):
    # Initialize lists to store values for all selected ROIs
    image_names = []
    r_list = []
    g_list = []
    b_list = []
    l_list = []
    a_list = []
    bb_list = []
    area_list = []
    perimeter_list = []
    equivalent_diameter_list = []
    bi_list = []
    elongation_list = []
    chroma_list = []
    hue_list = []
    roundness_list = []
    image_infos = []
    images_to_display = []  # List to store images in order
    threshold_values = []

    for idx, uploaded_file in enumerate(uploaded_files):
        # Read image
        image = Image.open(uploaded_file)
        image = np.array(image)

        image_info = int(uploaded_file.name.split(".")[0])
        image_infos.append(image_info)

        # Add a slider for threshold value
        st.header(f"Image: {uploaded_file.name}")
        threshold = st.slider(f"Set Threshold for {uploaded_file.name}", 0, 255, 128)

        # Process image
        processed_thresh = process_image(image, threshold)

        # Append grayscale image to the list
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract ROIs and contours
        rois, contours, masked_rois = extract_rois_from_contours(image, processed_thresh)
        rois_with_padding = add_padding_to_rois(rois, padding=10)

        # Morphological opening and closing images
        morph_open = cv2.morphologyEx(processed_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Append morphological opened and closed images to the list
        
        # Layout for displaying images
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(processed_thresh, caption="Processed Image (Morphological Ops)", use_column_width=True, clamp=True, channels="GRAY")

        with col2:
            if rois_with_padding:
                st.markdown("**ROIs with Separation and Selection**")

                # Create a dictionary to store ROI data
                roi_dict = {f"ROI {i+1}": roi for i, roi in enumerate(rois_with_padding)}

                # Display ROIs with a radio button for selection
                selected_roi_label = st.radio(
                    "Select an ROI for Processing:",
                    options=list(roi_dict.keys()),
                    key=f"radio_{uploaded_file.name}"  # Ensure unique key for each image
                )

                # Retrieve the selected ROI
                selected_roi = roi_dict[selected_roi_label]
                st.image(selected_roi, caption="Selected ROI", use_column_width=False)
                
                extracted_roi = masked_rois[list(roi_dict.keys()).index(selected_roi_label)]

                # Get the corresponding contour for the selected ROI
                selected_contour = contours[list(roi_dict.keys()).index(selected_roi_label)]

                # Calculate ROI properties
                area, perimeter, equivalent_diameter, r, g, b, l, a, bb, bi, chroma, hue, roundness, elongation = calculate_roi_properties(masked_rois[list(roi_dict.keys()).index(selected_roi_label)], selected_contour)
                
                
                # Convert the final image to LAB color space
                final_image_lab = cv2.cvtColor(extracted_roi, cv2.COLOR_BGR2LAB)

                # Split the LAB channels
                L_normalized, a_normalized, b_normalized = cv2.split(final_image_lab)
                a_normalized = cv2.normalize(a_normalized, None, 0, 255, cv2.NORM_MINMAX)
                b_normalized = cv2.normalize(b_normalized, None, 0, 255, cv2.NORM_MINMAX)
                
                if (idx==0):
                    images_to_display.append(image)
                    images_to_display.append(gray_image)
                    images_to_display.append(processed_thresh)
                    images_to_display.append(morph_open)
                    images_to_display.append(morph_close)
                    images_to_display.append(extracted_roi)
                    images_to_display.append(L_normalized)
                    images_to_display.append(a_normalized)
                    images_to_display.append(b_normalized)

                # Add the data to the lists
                # image_names.append(image_name)
                r_list.append(r)
                g_list.append(g)
                b_list.append(b)
                l_list.append(l)
                a_list.append(a)
                bb_list.append(bb)
                area_list.append(area)
                perimeter_list.append(perimeter)
                equivalent_diameter_list.append(equivalent_diameter)
                bi_list.append(bi)
                elongation_list.append(elongation)
                chroma_list.append(chroma)
                hue_list.append(hue)
                roundness_list.append(roundness)
                threshold_values.append(threshold)
    
    r_list = [int(r) for r in r_list]
    g_list = [int(g) for g in g_list]
    b_list = [int(b) for b in b_list]
    l_list = [float(l) for l in l_list]
    a_list = [float(a) for a in a_list]
    bb_list = [float(bb) for bb in bb_list]
    area_list = [float(area) for area in area_list]
    perimeter_list = [float(perimeter) for perimeter in perimeter_list]
    equivalent_diameter_list = [float(diameter) for diameter in equivalent_diameter_list]
    bi_list = [float(bi) for bi in bi_list]
    elongation_list = [float(elongation) for elongation in elongation_list]
    chroma_list = [float(chroma) for chroma in chroma_list]
    hue_list = [float(hue) for hue in hue_list]
    roundness_list = [float(roundness) for roundness in roundness_list]
    
    
    image_names.append("Reference Image")
    for i in range(1, len(r_list)):
        image_names.append(f"Image {i}")
        
    r_list, g_list, b_list, l_list, a_list, bb_list, area_list, perimeter_list, equivalent_diameter_list, bi_list, elongation_list, chroma_list, hue_list, roundness_list, image_infos = sorting_images(r_list, g_list, b_list, l_list, a_list, bb_list, area_list, perimeter_list, equivalent_diameter_list, bi_list, elongation_list, chroma_list, hue_list, roundness_list, image_infos)
    
    
    return image_names, r_list, g_list, b_list, l_list, a_list, bb_list, area_list, perimeter_list, equivalent_diameter_list, bi_list, elongation_list, chroma_list, hue_list, roundness_list, threshold_values, image_infos, images_to_display
