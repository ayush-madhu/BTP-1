import streamlit as st
import cv2
import numpy as np
from PIL import Image
from process_images import *


def sorting_images(thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos):
    
    # Choose the list to sort by (e.g., list1)
    combined = list(zip(image_infos, thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness))

    # Sort the combined list by the first element of each tuple (list1)
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Unzip the sorted lists back
    image_infos, thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness = zip(*sorted_combined)

    # Convert back to lists if needed
    image_infos = list(image_infos)
    thresholded_images = list(thresholded_images)
    image_names = list(image_names)
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
    
    return thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos

def sliders(uploaded_files):

    # Dictionary to store threshold values for each image
    threshold_values = {}

    if uploaded_files:
        st.write("Uploaded Images:")

        # Number of columns for layout
        num_columns = 3  # You can adjust this number based on how many columns you want to display
        cols = st.columns(num_columns)
        
        thresholded_images = []
        image_names = []
        avg_r = []
        avg_g = []
        avg_b = []
        avg_l = []
        avg_a = []
        avg_bb = []
        avg_area = []
        avg_perimeter = []
        avg_diameter = []
        avg_bi = []
        avg_chroma = []
        avg_hue = []
        avg_roundness = []
        image_infos = []
        avg_elongation = []

        # Step 2: Loop through the uploaded images and apply Otsu's thresholding
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            if (idx == 0):
                # image_name = f"Reference Image: {uploaded_file.name}"
                image_name = f"Reference Image"
            else:
                # image_name = f"Image {idx}: {uploaded_file.name}"
                image_name = f"Image {idx}"
            image_info = int(uploaded_file.name.split(".")[0])

            # Convert to grayscale if the image is in color
            grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)


            # Apply Triangle's thresholding as the initial threshold
            triangle_threshold_value, _ = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

            # Select the column to display the image in
            col_idx = idx % num_columns
            col = cols[col_idx]

            # Add a slider for threshold value below the image
            threshold_value = col.slider(
                f"Adjust Threshold for {image_name}",
                min_value=0,
                max_value=255,
                value=int(triangle_threshold_value),
                key=image_name
            )

            # Store threshold value in the dictionary
            threshold_values[image_name] = threshold_value

            # Apply the user's chosen threshold if the slider value is changed
            _, thresholded_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

            # Display the updated thresholded image
            col.image(thresholded_image, caption=f"Thresholded {image_name} (T={threshold_value})", use_column_width=True)
            
            # Add to the list of images to save
            thresholded_images.append((thresholded_image, uploaded_file.name, threshold_value))
            
            if (idx == 0):
                r, g, b, l, a, bb, area, perimeter, diameter, bi, elongation, chroma, hue, roundness, n, images_to_display = process_images(image, threshold_value, idx)
            else:
                r, g, b, l, a, bb, area, perimeter, diameter, bi, elongation, chroma, hue, roundness, _ = process_images(image, threshold_value, idx)
            
            image_names.append(image_name)
            avg_r.append(r)
            avg_g.append(g)
            avg_b.append(b)
            avg_l.append(l)
            avg_a.append(a)
            avg_bb.append(bb)
            avg_area.append(area)
            avg_perimeter.append(perimeter)
            avg_diameter.append(diameter)
            avg_bi.append(bi)
            avg_elongation.append(elongation)
            avg_chroma.append(chroma)
            avg_hue.append(hue)
            avg_roundness.append(roundness)
            image_infos.append(image_info)
        
        thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos = sorting_images(thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos)
            
    return thresholded_images, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos, n, images_to_display
