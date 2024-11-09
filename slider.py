import streamlit as st
import cv2
import numpy as np
from PIL import Image
from process_images import *


def sorting_images(
    image_names,
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
            image_names,
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
        image_names,
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

    return (
        image_names,
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

# Create a function to display the sliders and process the uploaded images
def sliders(uploaded_files):
    
    if uploaded_files:
        st.write("Uploaded Images:")

        # Step 1: Create a layout with multiple columns to display the images and sliders side by side
        num_columns = 3
        cols = st.columns(num_columns)

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
            if idx == 0:
                image_name = f"Reference Image"
            else:
                image_name = f"Image {idx}"
            image_info = int(uploaded_file.name.split(".")[0])

            # Convert to grayscale for thresholding and processing the image further
            grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # Apply Triangle's thresholding as the initial threshold value for the image (can be adjusted by the user) to create a binary image (black and white) for further processing and analysis of the image
            triangle_threshold_value, _ = cv2.threshold(
                grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
            )

            # Select the column to display the image in the layout based on the index of the image and the number of columns in the layout (num_columns) to arrange the images side by side
            col_idx = idx % num_columns
            col = cols[col_idx]

            # Add a slider for threshold value below the image to allow the user to adjust the threshold interactively for each image and see the effect on the binary image in real-time (if the slider value is changed)
            threshold_value = col.slider(
                f"Adjust Threshold for {image_name}",
                min_value=0,
                max_value=255,
                value=int(triangle_threshold_value),
                key=image_name,
            )

            # Apply the user's chosen threshold if the slider value is changed and create a binary image (black and white) based on the threshold value
            _, thresholded_image = cv2.threshold(
                grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV
            )

            # Display the updated thresholded image below the slider with the threshold value and the original image for comparison
            col.image(
                thresholded_image,
                caption=f"Thresholded {image_name} (T={threshold_value})",
                use_column_width=True,
            )
            
            # Process the uploaded image and extract the parameters for analysis
            if idx == 0:
                (
                    r,
                    g,
                    b,
                    l,
                    a,
                    bb,
                    area,
                    perimeter,
                    diameter,
                    bi,
                    elongation,
                    chroma,
                    hue,
                    roundness,
                    n,
                    images_to_display,
                ) = process_images(uploaded_file, threshold_value, idx)
            else:
                (
                    r,
                    g,
                    b,
                    l,
                    a,
                    bb,
                    area,
                    perimeter,
                    diameter,
                    bi,
                    elongation,
                    chroma,
                    hue,
                    roundness,
                    _,
                ) = process_images(uploaded_file, threshold_value, idx)

            # Append the processed image parameters to the respective lists for displaying in the final output table
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

        (
            image_names,
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
        ) = sorting_images(
            image_names,
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

    return (
        image_names,
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
        n,
        images_to_display,
    )
