import streamlit as st
import os
import cv2
import numpy as np
from skimage import measure
from PIL import Image
import pandas as pd
import time

import subprocess

try:
    import cv2
except ModuleNotFoundError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# Function to process the uploaded image
def process_image(image):
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Initialize RGB and LAB values lists, and lists for additional properties
    rgb_values = []
    lab_values = []
    centroids = []
    diameters = []
    perimeters = []
    areas = []

    images_to_display = []  # To store paths of extracted images
    processed_images = []   # To store paths of processed images (greyscale, binary, etc.)

    # Save the original image
    original_image_path = os.path.join(output_dir, 'original_image.png')
    # cv2.imwrite(original_image_path, image)
    processed_images.append(original_image_path)

    # Convert to greyscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image_path = os.path.join(output_dir, 'greyscale_image.png')
    # cv2.imwrite(grey_image_path, grey_image)
    processed_images.append(grey_image_path)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY_INV)
    binary_image_path = os.path.join(output_dir, 'binary_image.png')
    # cv2.imwrite(binary_image_path, binary_image)
    processed_images.append(binary_image_path)

    # Fill holes in the binary image
    filled_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    filled_image_path = os.path.join(output_dir, 'filled_image.png')
    # cv2.imwrite(filled_image_path, filled_image)
    processed_images.append(filled_image_path)

    # Apply morphological opening to remove small objects
    opened_image = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

    # Remove small objects (less than a certain area)
    labeled_image = measure.label(opened_image)
    regions = measure.regionprops(labeled_image)

    # Extract only the banana slices (regions in filled image that are white)
    count = 1
    for region in regions:
        if region.area >= 100:  # Threshold for area to filter out small noise
            minr, minc, maxr, maxc = region.bbox
            
            # Create a mask for the current region
            mask = np.zeros(filled_image.shape, dtype=np.uint8)
            mask[minr:maxr, minc:maxc] = (labeled_image[minr:maxr, minc:maxc] == region.label).astype(np.uint8)

            # Extract the region from the original image using the mask
            extracted_slice = cv2.bitwise_and(image, image, mask=mask)

            # Save the extracted slice
            extracted_slice_path = os.path.join(output_dir, f'extracted_slice_{count}.png')
            cv2.imwrite(extracted_slice_path, extracted_slice)
            images_to_display.append(extracted_slice_path)  # Save the path for later display
            count += 1

            # Calculate RGB and LAB values for the extracted slice
            avg_rgb = cv2.mean(extracted_slice, mask=mask)[:3]  # Get average RGB values
            lab_image = cv2.cvtColor(extracted_slice, cv2.COLOR_BGR2LAB)
            avg_lab = cv2.mean(lab_image, mask=mask)[:3]  # Get average LAB values

            # Calculate centroid, equivalent diameter, perimeter, and area
            centroid = region.centroid  # (y, x)
            eq_diameter = region.equivalent_diameter
            perimeter = region.perimeter
            area = region.area

            # Store the values
            rgb_values.append([int(round(val)) for val in avg_rgb])
            lab_values.append([int(round(val)) for val in avg_lab])
            centroids.append(centroid)
            diameters.append(eq_diameter)
            perimeters.append(perimeter)
            areas.append(area)

    return rgb_values, lab_values, centroids, diameters, perimeters, areas, images_to_display, processed_images

# Set the title of the app
st.markdown("<h1 style='text-align: center;'>Drying Analyzer</h1>", unsafe_allow_html=True)

# Create a hamburger menu using selectbox
option = st.selectbox(
    "Select Mode",
    ["Select Mode", "Static", "Dynamic"],
    index=0,
)

# Check the selected mode
if option == "Static":
    
    st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)
    
    # List of parameters with checkboxes
    parameters = [
        "Broning Index",
        "RGB",
        "LAB",
        "Centroid",
        "Equivalent Diameter",
        "Perimeter",
        "Area"
    ]
    
    define_all = st.checkbox('Select All')

    # Initialize the dictionary to store parameter selections
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state
    for option in parameters:
        selected_params[option] = st.checkbox(option, value=define_all)
    
    
    
    # # Master 'Select All' checkbox
    # define_all = st.checkbox('Select All')
    
    # # Generate checkboxes based on 'Select All' state
    # for option in parameters:
    #     st.checkbox(option, value=define_all)
    
    # selected_params = {param: st.checkbox(param, key=param) for param in parameters}
    
    # Centered button to open Google
    st.markdown("<div style='text-align: center;'><a href='https://www.google.com' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Open Camera</button></a></div>", unsafe_allow_html=True)
    
    # Add a gap below the "Open Camera" button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Adds a gap

    # Button to upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption='Uploaded Image', output_format='PNG')

        # Centered button to show results with the same style as "Open Camera"
        if st.button("Show Results", key="show_results"):
            # Process the uploaded image and get RGB, LAB values, and image paths
            rgb_values, lab_values, centroids, diameters, perimeters, areas, images_to_display, processed_images = process_image(image)

            # Prepare data dictionary based on selected parameters
            data = {}
            
            # Add Region column to the data
            data["Blob"] = [f"Blob {i + 1}" for i in range(len(rgb_values))]
            
            if selected_params.get("RGB"):
                data["RGB Values"] = [f"({r[0]}, {r[1]}, {r[2]})" for r in rgb_values]
            if selected_params.get("LAB"):
                data["LAB Values"] = [f"({l[0]}, {l[1]}, {l[2]})" for l in lab_values]
            if selected_params.get("Centroid"):
                data["Centroid"] = [f"({round(c[0], 2)}, {round(c[1], 2)})" for c in centroids]
            if selected_params.get("Equivalent Diameter"):
                data["Equivalent Diameter"] = [round(d, 2) for d in diameters]
            if selected_params.get("Perimeter"):
                data["Perimeter"] = [round(p, 2) for p in perimeters]
            if selected_params.get("Area"):
                data["Area"] = areas


            # Create a DataFrame to display results in a table
            results_df = pd.DataFrame(data)

            # Display the results in a table format
            st.markdown("<h3 style='text-align: center;'>Results:</h3>", unsafe_allow_html=True)
            st.dataframe(results_df.style.set_table_attributes('style="margin:auto; width:80%;"'))

            # Animate the images side by side
            st.markdown("<h3 style='text-align: center;'>Extracted and Processed Images:</h3>", unsafe_allow_html=True)
            
            # Create placeholders for extracted and processed images
            extracted_placeholder = st.empty()
            processed_placeholder = st.empty()
            
            # Number of times to repeat the animation
            repeat_count = 2
            
            for _ in range(repeat_count):
                # Display extracted images
                for image_path in images_to_display:
                    extracted_placeholder.image(image_path, use_column_width=True)  # Display the current extracted image
                    time.sleep(1)  # Wait for 1 second before displaying the next image
                
                # Display processed images
                for image_path in processed_images:
                    processed_placeholder.image(image_path, use_column_width=True)  # Display the current processed image
                    time.sleep(1)  # Wait for 1 second before displaying the next image


elif option == "Dynamic":
    st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)
    
    # List of parameters with checkboxes
    parameters = [
        "Broning Index",
        "RGB",
        "LAB",
        "Centroid",
        "Equivalent Diameter",
        "Perimeter",
        "Area"
    ]
    
    define_all = st.checkbox('Select All')

    # Initialize the dictionary to store parameter selections
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state
    for option in parameters:
        selected_params[option] = st.checkbox(option, value=define_all)
    
    # Centered button to open Google
    st.markdown("<div style='text-align: center;'><a href='https://www.google.com' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Take Reference Image</button></a></div>", unsafe_allow_html=True)
    
    # Add a gap below the "Take Reference Image" button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Adds a gap

    # Button to upload single image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption='Uploaded Image', output_format='PNG')

        # Show the new button after the image is uploaded
        st.markdown("<div style='text-align: center;'><a href='https://www.google.com' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Upload Image(s)</button></a></div>", unsafe_allow_html=True)
    
        # Add a gap below the "Take Reference Image" button
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Adds a gap


        # New button to upload multiple images
        multiple_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)

        if multiple_files is not None:
            st.markdown("<h3 style='text-align: center;'>Uploaded Images:</h3>", unsafe_allow_html=True)
            for uploaded in multiple_files:
                image = Image.open(uploaded)
                st.image(image, use_column_width=True, caption=uploaded.name, output_format='PNG')
