import streamlit as st
from PIL import Image
from image_processing import process_image
import pandas as pd
import time
from util import zip_images_and_dataframe

def static_mode():
    st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)

    # List of parameters with checkboxes
    parameters = [
    "Browning Index",
    "RGB",
    "L* a* b*",
    "Equivalent Diameter",
    "Perimeter",
    "Area",
    ]

    define_all = st.checkbox('Select All')

    # Initialize the dictionary to store parameter selections
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state
    for option in parameters:
        selected_params[option] = st.checkbox(option, value=define_all)

    # Centered button to open Google
    st.markdown("<div style='text-align: center;'><a href='http://192.168.253.102/' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Open Camera</button></a></div>", unsafe_allow_html=True)

    # Add a gap below the "Open Camera" button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Button to upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption='Uploaded Image', output_format='PNG')

        # Process the uploaded image and get RGB, LAB values, and image paths
        r, g, b, l, a, b, area, perimeter, diameter, bi, n, images_to_display = process_image(image)

        # Prepare data dictionary based on selected parameters
        data = {"Parameters" : [],
                "Values" : []}



        # Add Region column to the data

        if selected_params.get("RGB"):
            data['Parameters'].append('R')
            data['Parameters'].append('G')
            data['Parameters'].append('B')
            data['Values'].append(r)
            data['Values'].append(g)
            data['Values'].append(b)
            
        if selected_params.get("L* a* b*"):
            data['Parameters'].append('L*')
            data['Parameters'].append('a*')
            data['Parameters'].append('b*')
            data['Values'].append(l)
            data['Values'].append(a)
            data['Values'].append(b)
            
        if selected_params.get("Browning Index"):
            data['Parameters'].append('BI')
            data['Values'].append(bi) 
            
        if selected_params.get("Equivalent Diameter"):
            data['Parameters'].append('Diameter')
            data['Values'].append(diameter)
            
        if selected_params.get("Perimeter"):
            data['Parameters'].append('Perimeter')
            data['Values'].append(perimeter)
            
        if selected_params.get("Area"):
            data['Parameters'].append('Area')
            data['Values'].append(area)
                
        # Create a DataFrame to display results in a table
        results_df = pd.DataFrame(data)

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

        # Animate the images
        st.markdown("<h3 style='text-align: center;'>Image Processing Steps</h3>", unsafe_allow_html=True)

        # Create placeholders for extracted and processed images
        static_placeholder = st.empty()

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
        
        while (True):
            # Display extracted images
            count = 0
            for image_path in images_to_display:
                static_placeholder.image(image_path, caption=captions[count], use_column_width=True)
                count+=1# Display the current extracted image
                time.sleep(2)  # Wait for 1 second before displaying the next image