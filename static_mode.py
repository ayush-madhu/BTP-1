import streamlit as st
from PIL import Image
import pandas as pd
import time
import numpy as np
from util import create_zip
from image_processing import clear_output_directory, process_images
import cv2


# Function to display the static mode interface in Streamlit app
def static_mode():

    # Add a title to the app with markdown formatting and center alignment
    st.markdown(
        "<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True
    )

    # List of parameters with checkboxes to select the parameters to display in the results table and images to display in the app interface
    parameters = [
        "RGB",
        "L* a* b*",
        "Chroma",
        "Hue Angle",
        "Browning Index",
        "Equivalent Diameter",
        "Perimeter",
        "Area",
        "Roundness",
        "Elongation",
    ]

    # Checkbox to select all parameters at once
    define_all = st.checkbox("Select All")

    # Initialize the dictionary to store parameter selections with default values set to False for all parameters initially
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state to select the parameters to display in the results table and images to display in the app interface based on the user's selection
    for option in parameters:
        if option == "RGB":
            st.write("Colour Parameters")
        elif option == "Equivalent Diameter":
            st.write("Shape Parameters")
        elif option == "Roundness":
            st.write("Size Parameters")
        selected_params[option] = st.checkbox(option, value=define_all)

    # Add a markdown text to create space between the checkboxes and the upload button for better visibility
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Button to upload image files in the app interface with file types restricted to jpg, jpeg, and png formats and label visibility set to collapsed to hide the default label displayed by Streamlit for file upload buttons
    uploaded_file = st.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    # Check if an image file is uploaded by the user and proceed with image processing if an image is uploaded successfully in the app interface
    if uploaded_file is not None:
        clear_output_directory()
        # image = Image.open(uploaded_file)
        # image_np = np.array(image)
        # image_name = f"Image Name : {uploaded_file.name}"

        # # Convert the image to grayscale for processing using OpenCV functions and methods for image processing in Python with OpenCV library and convert the image to grayscale using the cv2.cvtColor() function with the cv2.COLOR_BGR2GRAY flag to convert the image from BGR to grayscale color space for processing the image in the app interface
        # grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # # Apply Triangle's thresholding as the initial threshold to the grayscale image using the cv2.threshold() function with the cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE flags to apply the Triangle's thresholding method to the grayscale image and get the threshold value and thresholded image for further processing in the app interface
        # triangle_threshold_value, _ = cv2.threshold(
        #     grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
        # )

        # # Add a slider for threshold value below the image to adjust the threshold value interactively in the app interface using the st.slider() function with the minimum and maximum values set to 0 and 255, respectively, and the initial value set to the threshold value calculated using Triangle's thresholding method for the grayscale image with the key set to the image name for tracking the slider value changes in the app interface and display the thresholded image based on the user's selected threshold value in the app interface
        # threshold_value = st.slider(
        #     f"Adjust Threshold for {image_name}",
        #     min_value=0,
        #     max_value=255,
        #     value=int(triangle_threshold_value),
        #     key=image_name,
        # )

        # # Apply the user's chosen threshold if the slider value is changed using the cv2.threshold() function with the threshold value selected by the user to get the thresholded image for further processing and display in the app interface based on the user's selected threshold value in the app interface for the uploaded image file
        # _, thresholded_image = cv2.threshold(
        #     grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV
        # )

        # # Display the updated thresholded image in the app interface with the caption showing the threshold value selected by the user for the uploaded image file in the app interface using the st.image() function with the thresholded image and caption as arguments to display the thresholded image with the threshold value caption below the image in the app interface for the user to visualize the thresholded image based on the selected threshold value in the app interface for the uploaded image file
        # st.image(
        #     thresholded_image,
        #     caption=f"Thresholded {image_name} (T={threshold_value})",
        #     use_column_width=True,
        # )

        # Process the uploaded image using the process_images() function to extract and process the regions of interest (ROIs) from the uploaded image based on the selected threshold value and parameters to display in the results table and images in the app interface for the user to analyze the image processing results interactively in the app interface
        (
            image_name,
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
            threshold_value,
            _,
            images_to_display,
        ) = process_images([uploaded_file])

        # Add Region column to the data frame to display the region number in the results table for the extracted regions from the uploaded image in the app interface for the user to analyze the extracted regions based on the selected parameters in the app interface
        data = {}
        data["Image No."] = image_name
        if selected_params.get("RGB"):
            data["R"] = r
            data["G"] = g
            data["B"] = b
        if selected_params.get("L* a* b*"):
            data["L*"] = l
            data["a*"] = a
            data["b*"] = bb
        if selected_params.get("Chroma"):
            data["Chroma"] = chroma
        if selected_params.get("Hue Angle"):
            data["Hue Angle"] = hue
        if selected_params.get("Equivalent Diameter"):
            data["Equivalent Diameter"] = diameter
        if selected_params.get("Perimeter"):
            data["Perimeter"] = perimeter
        if selected_params.get("Area"):
            data["Area"] = area
        if selected_params.get("Roundness"):
            data["Roundness"] = roundness
        if selected_params.get("Elongation"):
            data["Elongation"] = elongation
        if selected_params.get("Browning Index"):
            data["BI"] = bi

        # Create a DataFrame to display results in a table format in the app interface using the pd.DataFrame() function with the data dictionary as input to create a DataFrame with the extracted image processing results for the uploaded image in the app interface
        data["Threshold Value"] = threshold_value
        results_df = pd.DataFrame(data=data, index=[0]).T.reset_index()
        results_df.columns = ["Parameters", "Value"]

        if st.button("OK"):

            # Add a title to the app with markdown formatting and center alignment
            st.markdown(
                "<h3 style='text-align: center;'>Image Processing Steps</h3>",
                unsafe_allow_html=True,
            )

            # Create placeholders for extracted and processed images to display in the app interface
            static_placeholder = st.empty()

            # Define CSS to center-align the table in the app interface
            st.markdown(
                "<h3 style='text-align: center;'>Results</h3>", unsafe_allow_html=True
            )

            # Create an HTML table from the DataFrame to display the results in a tabular format in the app interface
            table_html = results_df.to_html(index=False)

            # Center-align the table using a styled div in HTML to center the table in the app interface for better visibility and readability of the results
            centered_table_html = f"""
            <div style="display: flex; justify-content: center;">
                {table_html}
            </div>
            """

            # Render the centered table in Streamlit
            st.markdown(centered_table_html, unsafe_allow_html=True)

            # Define captions for the extracted images to display in the app interface for the user to visualize the image processing steps and results interactively in the app interface
            captions = [
                "Orginial Image",
                "Greyscale Image",
                "Triangular Thresholding",
                "Morphological Opening",
                "Morphological Closing",
                "Extracted ROI",
                "L Channel",
                "a Channel",
                "b Channel"
            ]

            # All the data are stored in the output folder and zipped to download the data
            zip_data = create_zip(images_to_display,_, results_df)

            # Add a download button to download the extracted images and results as a ZIP file in the app interface for the user to download the processed images and results for further analysis and sharing
            st.download_button(
                label="Download Data (ZIP)",
                data=zip_data,
                file_name="Static Data.zip",
                mime="application/zip",
            )
            
            # Display extracted images and processed images in the app interface using the static_placeholder.empty() function to create a placeholder for displaying the images and the static_placeholder.image() function to display the images in the placeholder with the image path and caption as arguments to display the images with captions in the app interface for the user to visualize the image processing steps and results interactively
            while True:
                count = 0
                for i in images_to_display:
                    static_placeholder.image(
                        i, caption=captions[count], use_column_width=True
                    )
                    count += 1
                    time.sleep(2)
