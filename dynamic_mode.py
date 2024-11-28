import streamlit as st
import pandas as pd
import time
from util import create_zip
import numpy as np
import matplotlib.pyplot as plt
from image_processing import process_images, clear_output_directory
import io

# Function to calculate the time difference in seconds between two timestamps (t1 and t2) in the format 'YYYYMMDDHHMMSS' (e.g., '20220101120000') and return the difference in seconds as an integer
def time_difference_in_seconds(t1, t2):
    year_diff = (int(t2[:4]) - int(t1[:4])) * 365 * 24 * 3600  # Year difference
    month_diff = (
        (int(t2[4:6]) - int(t1[4:6])) * 30 * 24 * 3600
    )  # Month difference (approx 30 days)
    day_diff = (int(t2[6:8]) - int(t1[6:8])) * 24 * 3600  # Day difference
    hour_diff = (int(t2[8:10]) - int(t1[8:10])) * 3600  # Hour difference
    minute_diff = (int(t2[10:12]) - int(t1[10:12])) * 60  # Minute difference
    second_diff = int(t2[12:14]) - int(t1[12:14])  # Second difference

    return year_diff + month_diff + day_diff + hour_diff + minute_diff + second_diff


# Function to display the dynamic mode interface for selecting parameters and uploading multiple images for processing and analysis
def dynamic_mode():
    st.markdown(
        "<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True
    )

    # List of parameters with checkboxes for selection in the dynamic mode interface
    parameters = [
        "RGB",
        "L* a* b*",
        "∆E",
        "Browning Index",
        "Chroma",
        "Hue Angle",
        "Roundness",
        "Elongation",
        "Equivalent Diameter",
        "Perimeter",
        "Area",
        "Shrinkage"
    ]

    define_all = st.checkbox("Select All")

    # Initialize the dictionary to store parameter selections with default values set to False for each parameter in the list of parameters defined above (parameters)
    selected_params = {param: False for param in parameters}

    # Generate checkboxes based on 'Select All' state and store the selected parameters in the selected_params dictionary with True values for the selected parameters and False for the unselected parameters
    for option in parameters:
        if option == "RGB":
            st.write("Colour Parameters")
        elif option == "Roundness":
            st.write("Shape Parameters")
        elif option == "Equivalent Diameter":
            st.write("Size Parameters")
        selected_params[option] = st.checkbox(option, value=define_all)

    # New button to upload multiple images for processing and analysis
    multiple_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Check if the user has uploaded multiple images and display the sliders for setting the threshold values for each image
    if multiple_files:
        clear_output_directory()
        
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
            threshold_values,
            image_infos,
            images_to_display,
        ) = process_images(multiple_files)
        
        if st.button("Done"):
            reference = str(image_infos[0])
            image_infos = [str(i) for i in image_infos]
            time_differences = [
                time_difference_in_seconds(reference, f) / 3600 for f in image_infos
            ] # Calculate time differences in hours
            
            check = 0
            row = {}
            row["Image No."] = image_names
            row["Threshold Values"] = threshold_values
            if selected_params.get("RGB"):
                row["R"] = avg_r
                row["G"] = avg_g
                row["B"] = avg_b
            if selected_params.get("L* a* b*"):
                row["L*"] = avg_l
                row["a*"] = avg_a
                row["b*"] = avg_bb
                check = 1
            if selected_params.get("Browning Index"):
                row["BI"] = avg_bi
                check = 1
            if selected_params.get("∆E"):
                avg_l = np.array(avg_l)
                avg_a = np.array(avg_a)
                avg_bb = np.array(avg_bb)
                row["∆E"] = (
                    np.round(
                        np.sqrt(
                            (avg_l[0] - avg_l) ** 2
                            + (avg_a[0] - avg_a) ** 2
                            + (avg_bb[0] - avg_bb) ** 2
                        ),
                        2,
                    )
                ).tolist()
                avg_l = avg_l.tolist()
                avg_a = avg_a.tolist()
                avg_bb = avg_bb.tolist()
                check = 1
            if selected_params.get("Chroma"):
                row["Chroma"] = avg_chroma
            if selected_params.get("Hue Angle"):
                row["Hue Angle"] = avg_hue
            if selected_params.get("Roundness"):
                row["Roundness"] = avg_roundness
            if selected_params.get("Elongation"):
                row["Elongation"] = avg_elongation
            if selected_params.get("Equivalent Diameter"):
                row["Equivalent Diameter"] = avg_diameter
            if selected_params.get("Perimeter"):
                row["Perimeter"] = avg_perimeter
            if selected_params.get("Area"):
                row["Area"] = avg_area
            if selected_params.get("Shrinkage"):
                row["Shrinkage"] = [((avg_area[0] - x) / avg_area[0])*100 for x in avg_area]
                row["Shrinkage"] = [round(x, 2) for x in row["Shrinkage"]]
            
            results_df_avg = pd.DataFrame(row)

            # Plot the selected parameters based on the time differences between the images (in hours) using matplotlib and save the plots as images in the 'output' folder for display
            
            figures = []
            
            if selected_params.get("Browning Index"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    results_df_avg["BI"],
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Browning Index",
                )
                ax.set_title("Browning Index Plot")
                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Average BI")
                ax.legend(loc="upper right")
                figures.append(fig)
                
            if selected_params.get("L* a* b*"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    results_df_avg["L*"],
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="L*",
                )
                ax.plot(
                    time_differences,
                    results_df_avg["a*"],
                    marker="^",
                    markersize=5,
                    color="red",
                    label="a*",
                )
                ax.plot(
                    time_differences,
                    results_df_avg["b*"],
                    marker="d",
                    markersize=5,
                    color="green",
                    label="b*",
                )
                ax.set_title("L*a*b* Colour")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Average Color")
                ax.legend(loc="upper right")
                figures.append(fig)

            if selected_params.get("∆E"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    results_df_avg["∆E"],
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="∆E",
                )
                ax.set_title("∆E")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Colour Difference (∆E)")
                ax.legend(loc="upper right")
                figures.append(fig)
            
            avg_area = np.array(avg_area)
            area_diff = (avg_area[0]-avg_area).tolist()
            if selected_params.get("Area"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    area_diff,
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Area",
                )
                ax.set_title("Area")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Change in Area")
                ax.legend(loc="upper right")
                figures.append(fig)
            
            avg_roundness = np.array(avg_roundness)
            roundness_diff = (avg_roundness[0]-avg_roundness).tolist()
            if selected_params.get("Roundness"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    roundness_diff,
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Roundness",
                )
                ax.set_title("Roundness")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Change in Roundness")
                ax.legend(loc="upper right")
                figures.append(fig)
                
            avg_elongation = np.array(avg_elongation)
            elongation_diff = (avg_elongation[0]-avg_elongation).tolist()
            if selected_params.get("Roundness"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    elongation_diff,
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Elongation",
                )
                ax.set_title("Elongation")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Change in Elongation")
                ax.legend(loc="upper right")
                figures.append(fig)
            
            avg_perimeter = np.array(avg_perimeter)   
            perimeter_diff = (avg_perimeter[0]-avg_perimeter).tolist()
            if selected_params.get("Roundness"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    perimeter_diff,
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Perimeter",
                )
                ax.set_title("Perimeter")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Change in Perimeter")
                ax.legend(loc="upper right")
                figures.append(fig)
                
            shrinkage = ((avg_area[0]-avg_area)/avg_area[0]).tolist()  
            shrinkage = np.array(shrinkage)              
            shrinkage_diff = (np.round(shrinkage[0]-shrinkage,2)).tolist()
            if selected_params.get("Shrinkage"):
                fig, ax = plt.subplots()
                ax.plot(
                    time_differences,
                    shrinkage_diff,
                    marker="o",
                    markersize=5,
                    color="blue",
                    label="Shrinkage",
                )
                ax.set_title("Shrinkage")
                ax.set_xlabel("Time, hours")
                ax.set_ylabel("Change in Shrinkage")
                ax.legend(loc="upper right")
                figures.append(fig)
                
                

            # Add heading for the image processing steps section
            st.markdown(
                "<h3 style='text-align: center;'>Image Processing Steps</h3>",
                unsafe_allow_html=True,
            )

            # Create placeholders for extracted and processed images and display the images in the placeholders
            dynamic_placeholder = st.empty()

            # Define the captions for the images to be displayed in the placeholders based on the image processing steps
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
            # Display the results in a table format
            st.markdown(
                "<h2 style='text-align: center;'>Results</h2>", unsafe_allow_html=True
            )
            st.dataframe(results_df_avg)

            # Display the graphs based on the selected parameters
            if check == 1:
                st.markdown(
                    "<h2 style='text-align: center;'>Graphs</h2>",
                    unsafe_allow_html=True,
                )

            if selected_params.get("Browning Index"):
                graph1_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[0].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph1_placeholder.image(
                    img_buffer,
                    caption="Browning Index Plot",
                    use_column_width=True,
                )
            if selected_params.get("L* a* b*"):
                graph2_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[1].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph2_placeholder.image(
                    img_buffer, caption="L*a*b* Plot", use_column_width=True
                )
            if selected_params.get("∆E"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[2].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆E Plot", use_column_width=True
                )
                
            if selected_params.get("Area"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[3].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆Area Plot", use_column_width=True
                )
                
            if selected_params.get("Roundness"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[4].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆Roundness Plot", use_column_width=True
                )
                
            if selected_params.get("Elongation"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[5].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆Elongation Plot", use_column_width=True
                )
                
            if selected_params.get("Perimeter"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[6].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆Perimeter Plot", use_column_width=True
                )
                
            if selected_params.get("Shrinkage"):
                graph3_placeholder = st.empty()
                img_buffer = io.BytesIO()
                figures[7].savefig(img_buffer, format="PNG")  # Save the figure as PNG into the buffer
                img_buffer.seek(0)
                graph3_placeholder.image(
                    img_buffer, caption="∆Shrinkage Plot", use_column_width=True
                )
            
            zip_file = create_zip(images_to_display, figures, results_df_avg)
            st.download_button(
                label="Download ZIP",
                data=zip_file.getvalue(),
                file_name="output.zip",
                mime="application/zip"
            )

            # Image Prcoessing Steps Display Loop for Dynamic Mode
            while True:
                count = 0
                for i in images_to_display:
                    dynamic_placeholder.image(
                        i, caption=captions[count], use_column_width=True
                    )
                    count += 1
                    time.sleep(2)
