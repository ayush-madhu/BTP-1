import streamlit as st
from PIL import Image
# from image_processing import process_image
import pandas as pd
import time
from util import zip_images_and_dataframe
import numpy as np
import matplotlib.pyplot as plt
from slider import *
import os

def time_difference_in_seconds(t1, t2):
    year_diff = (int(t2[:4]) - int(t1[:4])) * 365 * 24 * 3600  # Year difference
    month_diff = (int(t2[4:6]) - int(t1[4:6])) * 30 * 24 * 3600  # Month difference (approx 30 days)
    day_diff = (int(t2[6:8]) - int(t1[6:8])) * 24 * 3600  # Day difference
    hour_diff = (int(t2[8:10]) - int(t1[8:10])) * 3600  # Hour difference
    minute_diff = (int(t2[10:12]) - int(t1[10:12])) * 60  # Minute difference
    second_diff = int(t2[12:14]) - int(t1[12:14])  # Second difference
    
    return year_diff + month_diff + day_diff + hour_diff + minute_diff + second_diff

# Function to display the dynamic mode interface
def dynamic_mode():
    st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)
            
    # List of parameters with checkboxes
    parameters = [
    "RGB",
    "L* a* b*",
    "∆E",
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

    # New button to upload multiple images
    multiple_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)

    # Table Making
    if multiple_files:
        
        _, image_names, avg_r, avg_g, avg_b, avg_l, avg_a, avg_bb, avg_area, avg_perimeter, avg_diameter, avg_bi, avg_elongation, avg_chroma, avg_hue, avg_roundness, image_infos, n, images_to_display = sliders(multiple_files)
        if st.button('OK'):
            
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            reference = str(image_infos[0])
            image_infos = [str(i) for i in image_infos] # Convert to string for comparison
            time_differences = [time_difference_in_seconds(reference, f) / 3600 for f in image_infos]

            check = 0
            row = {}
            row['Image No.'] = image_names
            if selected_params.get("RGB"):
                row['R'] = avg_r
                row['G'] = avg_g
                row['B'] = avg_b
            if selected_params.get("L* a* b*"):
                row['L*'] = avg_l
                row['a*'] = avg_a
                row['b*'] = avg_bb
                check = 1
            if selected_params.get("∆E"):
                avg_l = np.array(avg_l)
                avg_a = np.array(avg_a)
                avg_bb = np.array(avg_bb)
                row['∆E'] = (np.round(np.sqrt((avg_l[0] - avg_l)**2 + (avg_a[0] - avg_a)**2 + (avg_bb[0] - avg_bb)**2), 2)).tolist()
                avg_l = avg_l.tolist()
                avg_a = avg_a.tolist()
                avg_bb = avg_bb.tolist()
                check = 1
            if selected_params.get("Chroma"):
                row['Chroma'] = avg_chroma
            if selected_params.get("Hue Angle"):
                row['Hue Angle'] = avg_hue
            if selected_params.get("Equivalent Diameter"):
                row['Equivalent Diameter'] = avg_diameter
            if selected_params.get("Perimeter"):
                row['Perimeter'] = avg_perimeter
            if selected_params.get("Area"):
                row['Area'] = avg_area
            if selected_params.get("Roundness"):
                row['Roundness'] = avg_roundness
            if selected_params.get("Elongation"):
                row['Elongation'] = avg_elongation
            if selected_params.get("Browning Index"):
                row['BI'] = avg_bi
                check = 1
            
            results_df_avg = pd.DataFrame(row)

            if selected_params.get("Browning Index"):
                plt.figure()
                plt.plot(time_differences, results_df_avg['BI'], marker='o', markersize=5, color='blue', label='Browning Index')
                plt.title('Browning Index Plot')
                plt.xlabel('Time (hours)')
                plt.ylabel('Average BI')
                plt.legend(loc = 'upper right')
                plt.savefig('output/bi_plot.png')

            if selected_params.get("L* a* b*"):
                plt.figure()
                plt.plot(time_differences, results_df_avg['L*'], marker='o', markersize=5, color='blue', label='L*')
                plt.plot(time_differences, results_df_avg['a*'], marker='^', markersize=5, color='red', label='a*')
                plt.plot(time_differences, results_df_avg['b*'], marker='d', markersize=5, color='green', label='b*')
                plt.title('L*a*b* Colour')
                plt.xlabel('Time, hours')
                plt.ylabel('Average Color')
                plt.legend(loc = 'upper right')
                plt.savefig('output/lab_plot.png')

            if selected_params.get("∆E"):
                plt.figure()
                plt.plot(time_differences, results_df_avg['∆E'], marker='o', markersize=5, color='blue', label='∆E')
                plt.title('∆E')
                plt.xlabel('Time, hours')
                plt.ylabel('Colour Difference (∆E)')
                plt.legend(loc = 'upper right')
                plt.savefig('output/dele_plot.png')
                
                
            # Animate the images side by side
            st.markdown("<h3 style='text-align: center;'>Image Processing Steps</h3>", unsafe_allow_html=True)

            # Create placeholders for extracted and processed images
            dynamic_placeholder = st.empty()

            caption1 = ['Orginial Image', 'Greyscale Image', 'Triangular Thresholding', 'Morphological Opening', 'Morphological Closing', 'Extracted Regions']
            
            caption2 = [f'ROI {i+1}' for i in range(0, n)]
            caption3 = ['L Channel', 'a Channel', 'b Channel']

            captions = caption1 + caption2 + caption3

            # Display the results in a table
            st.markdown("<h2 style='text-align: center;'>Results</h2>", unsafe_allow_html=True)
            st.dataframe(results_df_avg)

            if (check == 1):
                st.markdown("<h2 style='text-align: center;'>Graphs</h2>", unsafe_allow_html=True)

            if selected_params.get("Browning Index"):
                graph1_placeholder = st.empty()
                graph1_placeholder.image('output/bi_plot.png', caption='Browning Index Plot', use_column_width=True)
            if selected_params.get("L* a* b*"):
                graph2_placeholder = st.empty()
                graph2_placeholder.image('output/lab_plot.png', caption='L*a*b* Plot' ,use_column_width=True)
            if selected_params.get("∆E"):
                graph3_placeholder = st.empty()
                graph3_placeholder.image('output/dele_plot.png', caption='Del E Plot' ,use_column_width=True)

            image_folder = 'output'  # Update this path to your folder containing the images

            zip_data = zip_images_and_dataframe(image_folder, results_df_avg)

            st.download_button(
            label="Download Data (ZIP)",
            data=zip_data,
            file_name="Dynamic Data.zip",
            mime="application/zip"
            )
            
            while (True):
                # Image Prcoessing Animation
                count = 0
                for i in images_to_display:
                    dynamic_placeholder.image(i, caption=captions[count], use_column_width=True)
                    count+=1
                    time.sleep(2)