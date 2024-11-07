import streamlit as st
from PIL import Image
import pandas as pd
import time
from util import zip_images_and_dataframe
from slider import *
import cv2
import shutil

def static_mode():
    
    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)

    # List of parameters with checkboxes
    parameters = [
    "RGB",
    "L* a* b*",
    "Chroma",
    "Hue Angle",
    "Equivalent Diameter",
    "Perimeter",
    "Area",
    "Roundness",
    "Elongation",
    "Browning Index"
    ]

    define_all = st.checkbox('Select All')

    # Initialize the dictionary to store parameter selections
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state
    for option in parameters:
        if (option == 'RGB'):
            st.write('Colour Parameters')
        elif (option == 'Equivalent Diameter'):
            st.write('Shape Parameters')
        selected_params[option] = st.checkbox(option, value=define_all)

    # Add a gap below the "Open Camera" button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Button to upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_name = f"Image Name : {uploaded_file.name}"

        # Convert to grayscale if the image is in color
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply Triangle's thresholding as the initial threshold
        triangle_threshold_value, _ = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

        # Add a slider for threshold value below the image
        threshold_value = st.slider(
            f"Adjust Threshold for {image_name}",
            min_value=0,
            max_value=255,
            value=int(triangle_threshold_value),
            key=image_name
        )

        # Apply the user's chosen threshold if the slider value is changed
        _, thresholded_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Display the updated thresholded image
        st.image(thresholded_image, caption=f"Thresholded {image_name} (T={threshold_value})", use_column_width=True)
        
        r, g, b, l, a, bb, area, perimeter, diameter, bi, elongation, chroma, hue, roundness, n, images_to_display = process_images(image_np, threshold_value, 0)

        # Add Region column to the data
        data = {}
        data['Image No.'] = image_name
        if selected_params.get("RGB"):
            data['R'] = r
            data['G'] = g
            data['B'] = b
        if selected_params.get("L* a* b*"):
            data['L*'] = l
            data['a*'] = a
            data['b*'] = bb
        if selected_params.get("Chroma"):
            data['Chroma'] = chroma
        if selected_params.get("Hue Angle"):
            data['Hue Angle'] = hue
        if selected_params.get("Equivalent Diameter"):
            data['Equivalent Diameter'] = diameter
        if selected_params.get("Perimeter"):
            data['Perimeter'] = perimeter
        if selected_params.get("Area"):
            data['Area'] = area
        if selected_params.get("Roundness"):
            data['Roundness'] = roundness
        if selected_params.get("Elongation"):
            data['Elongation'] = elongation
        if selected_params.get("Browning Index"):
            data['BI'] = bi
                
        # Create a DataFrame to display results in a table
        results_df = pd.DataFrame(data=data, index=[0]).T.reset_index()
        results_df.columns = ["Parameters","Value"]
        # results_df = results_df.T

        if (st.button("OK")):
            
            # Animate the images
            st.markdown("<h3 style='text-align: center;'>Image Processing Steps</h3>", unsafe_allow_html=True)

            # Create placeholders for extracted and processed images
            static_placeholder = st.empty()
            
            # Define CSS to center-align the table
            st.markdown("<h3 style='text-align: center;'>Results</h3>", unsafe_allow_html=True) 
            
            # Create an HTML table from the DataFrame
            table_html = results_df.to_html(index=False)

            # Center-align the table using a styled div
            centered_table_html = f"""
            <div style="display: flex; justify-content: center;">
                {table_html}
            </div>
            """

            # Render the centered table in Streamlit
            st.markdown(centered_table_html, unsafe_allow_html=True)

            caption1 = ['Orginial Image', 'Greyscale Image', 'Triangular Thresholding', 'Morphological Opening', 'Morphological Closing', 'Extracted Regions']
            caption2 = [f'ROI {i+1}' for i in range(0, n)]
            caption3 = ['L Channel', 'a Channel', 'b Channel']

            captions = caption1 + caption2 + caption3
            
            zip_data = zip_images_and_dataframe('output', results_df)

            st.download_button(
                label="Download Data (ZIP)",
                data=zip_data,
                file_name="Static Data.zip",
                mime="application/zip"
            )
            
            # Display extracted images
            while (True):
                count = 0
                for image_path in images_to_display:
                    static_placeholder.image(image_path, caption=captions[count], use_column_width=True)
                    count+=1
                    time.sleep(2)