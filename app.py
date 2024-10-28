import streamlit as st
import os
import cv2
import numpy as np
from skimage import measure
from PIL import Image
import pandas as pd
import time
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import shutil





# Function to process the uploaded image
def process_image(image):
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
    images_to_display = []

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # Load the image in BGR format
    image_original_path = os.path.join(output_dir, 'original_image.jpg')
    cv2.imwrite(image_original_path, image_bgr)
    images_to_display.append(image_original_path)

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

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray_path = os.path.join(output_dir, 'gray_image.jpg')
    cv2.imwrite(image_gray_path, image_gray)
    images_to_display.append(image_gray_path)

    _,triangle = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)
    triangle_path = os.path.join(output_dir, 'triangle.jpg')
    cv2.imwrite('output/triangle.jpg', triangle)
    images_to_display.append(triangle_path)

    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(triangle, cv2.MORPH_OPEN, kernel)
    opening_path = os.path.join(output_dir, 'opening.jpg')
    cv2.imwrite(opening_path, opening)
    images_to_display.append(opening_path)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing_path = os.path.join(output_dir, 'closing.jpg')
    cv2.imwrite(closing_path, closing)
    images_to_display.append(closing_path)

    extraction = cv2.bitwise_and(image_bgr, image_bgr, mask=closing)
    extraction_path = os.path.join(output_dir, 'extraction.jpg')
    cv2.imwrite(extraction_path, extraction)
    images_to_display.append(extraction_path)

    # Find contours of the ROIs
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the individual masked ROIs
    roi_rgb = []
    roi_lab = []
    roi_bi = []
    roi_diameter = []
    roi_perimeter = []
    roi_area = []
    roi_centroid = []

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
        
        roi_path = os.path.join(output_dir, f'roi_{i+1}.jpg')
        cv2.imwrite(roi_path, roi)
        images_to_display.append(roi_path)
        
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

        # Centroid
        centroid = region.centroid  # (row, column)

        roi_area.append(round(area))
        roi_perimeter.append(round(perimeter))
        roi_diameter.append(round(diameter))
        roi_centroid.append((round(centroid[0]), round(centroid[1])))
        
    extraction_lab = cv2.cvtColor(extraction, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, a, and b channels
    L, a, b = cv2.split(extraction_lab)

    # Optional: Normalize the L, a, b channels for proper visualization
    # L is already in the range 0-100 (usually), but a and b need to be adjusted from -128 to 127 to 0-255 for viewing.
    L_normalized = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_normalized = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)

    # Save the L, a, and b channel images individually
    L_image_path = os.path.join(output_dir, 'L_image.jpg')
    cv2.imwrite(L_image_path, L_normalized)
    images_to_display.append(L_image_path)
    a_image_path = os.path.join(output_dir, 'a_image.jpg')
    cv2.imwrite(a_image_path, a_normalized)
    images_to_display.append(a_image_path)
    b_image_path = os.path.join(output_dir, 'b_image.jpg')
    cv2.imwrite(b_image_path, b_normalized)
    images_to_display.append(b_image_path)
        
    images_to_display = [s.replace('\\', '/') for s in images_to_display]

    return roi_rgb, roi_lab, roi_centroid, roi_diameter, roi_perimeter, roi_area, roi_bi, images_to_display





def zip_images_and_dataframe(image_folder, dataframe):
    zip_buffer = BytesIO()  # Create a BytesIO buffer to hold the zip file data
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        # Add images from the folder to the zip file
        for root, _, files in os.walk(image_folder):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as img_file:
                    zf.writestr(file, img_file.read())

        # Convert DataFrame to Excel and add it to the zip file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False)
        excel_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        zf.writestr('table_data.xlsx', excel_buffer.getvalue())
    
    zip_buffer.seek(0)  # Move the cursor to the beginning of the buffer
    return zip_buffer






# Program starts here
if __name__ == "__main__":

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
            "Browning Index",
            "RGB",
            "L* a* b*",
            "Centroid",
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
        st.markdown("<div style='text-align: center;'><a href='https://www.google.com' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Open Camera</button></a></div>", unsafe_allow_html=True)
        
        # Add a gap below the "Open Camera" button
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Adds a gap
            
        
        

        # Button to upload image
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption='Uploaded Image', output_format='PNG')

            # Process the uploaded image and get RGB, LAB values, and image paths
            rgb_values, lab_values, centroids, diameters, perimeters, areas, bi, images_to_display = process_image(image)

            # Prepare data dictionary based on selected parameters
            data = {}
            
            # Add Region column to the data
            data["Blob"] = [f"Blob {i + 1}" for i in range(len(rgb_values))]
            
            if selected_params.get("RGB"):
                data['R'] = [r[0] for r in rgb_values]
                data['G'] = [r[1] for r in rgb_values]
                data['B'] = [r[2] for r in rgb_values]
            if selected_params.get("L*a*b*"):
                data['L*'] = [l[0] for l in lab_values]
                data['a*'] = [l[1] for l in lab_values]
                data['b*'] = [l[2] for l in lab_values]
            if selected_params.get("Browning Index"):
                data["Browning Index"] = bi
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
            df = results_df.set_index('Blob')

            # Display the results in a table format
            st.markdown("<h3 style='text-align: center;'>Results</h3>", unsafe_allow_html=True)
            st.dataframe(results_df.style.set_table_attributes('style="margin:auto; width:80%;"'))

            # Animate the images side by side
            st.markdown("<h3 style='text-align: center;'>Image Processing Steps</h3>", unsafe_allow_html=True)
            
            # Create placeholders for extracted and processed images
            static_placeholder = st.empty()
            
            caption1 = ['Orginial Image', 'Greyscale Image', 'Triangular Thresholding', 'Morphological Opening', 'Morphological Closing', 'Extracted Regions']
            caption2 = [f'ROI {i+1}' for i in range(0, len(rgb_values))]
            caption3 = ['L Channel', 'a Channel', 'b Channel']
            
            captions = caption1 + caption2 + caption3
            
            zip_data = zip_images_and_dataframe('output', df)

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
                    time.sleep(3)  # Wait for 1 second before displaying the next image
                











    elif option == "Dynamic":
        
        st.markdown("<h2 style='text-align: center;'>Select Parameters</h2>", unsafe_allow_html=True)
        
        # List of parameters with checkboxes
        parameters = [
            "Browning Index",
            "RGB",
            "L* a* b*",
            "Centroid",
            "Equivalent Diameter",
            "Perimeter",
            "Area",
            "∆E"
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
            st.image(image, use_column_width=True, caption='Uploaded Reference Image', output_format='PNG')

            # Show the new button after the image is uploaded
            st.markdown("<div style='text-align: center;'><a href='https://www.google.com' target='_blank' style='font-size: 18px; text-decoration: none;'><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Upload Image(s)</button></a></div>", unsafe_allow_html=True)
        
            # Add a gap below the "Take Reference Image" button
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Adds a gap

            # New button to upload multiple images
            multiple_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)
            
            # Process the uploaded Reference image and get RGB, LAB values, and image paths
            r_rgb_values, r_lab_values, r_centroids, r_diameters, r_perimeters, r_areas, r_bi, r_images_to_display = process_image(image)
            
                
            # Function to process multiple images and display results in a table
            def process_images(images):
                data = []
                
                for idx, image_file in enumerate(images):
                    image = Image.open(image_file)
                    rgb_values, lab_values, centroids, diameters, perimeters, areas, bi, _ = process_image(image)

                    image_label = f"{idx + 1}"  # Sequential naming for images
                    
                    for i in range(len(rgb_values)):
                        row = {}
                        row['Image No.'] = image_label
                        row['Blob'] = f'{i + 1}'
                        if selected_params.get("RGB"):
                            row['R'] = rgb_values[i][0]
                            row['G'] = rgb_values[i][1]
                            row['B'] = rgb_values[i][2]
                        if selected_params.get("L* a* b*"):
                            row['L*'] = lab_values[i][0]
                            row['a*'] = lab_values[i][1]
                            row['b*'] = lab_values[i][2]
                        if selected_params.get("Browning Index"):
                            row['BI'] = bi[i]
                        if selected_params.get("Centroid"):
                            row['x'] = int(round(centroids[i][1]))
                            row['y'] = int(round(centroids[i][0]))
                        if selected_params.get("Equivalent Diameter"):
                            row['Equivalent Diameter'] = diameters[i]
                        if selected_params.get("Perimeter"):
                            row['Perimeter'] = perimeters[i]
                        if selected_params.get("Area"):
                            row['Area'] = areas[i]
                        #if selected_params.get("∆E"):
                           # row['∆E'] = round(np.sqrt((r_lab_values[i][0] - lab_values[i][0]) ** 2 + (r_lab_values[i][1] - lab_values[i][1]) ** 2 + (r_lab_values[i][2] - lab_values[i][2]) ** 2))
                            
                        data.append(row)
                
                return pd.DataFrame(data)

            # Table Making
            if multiple_files:
                
                multiple_files_names = [int(f.name.split('.')[0]) for f in multiple_files]
                uploaded_file_name = int(uploaded_file.name.split('.')[0])
                all_files = [uploaded_file] + multiple_files
                all_files_names = [uploaded_file_name] + multiple_files_names
                
                # Combine the lists with their indices
                combined = [(index, all_files[index], all_files_names[index]) for index in range(len(all_files))]

                # Sort combined by the second element of the tuple (which is List2 values)
                sorted_combined = sorted(combined, key=lambda x: x[2])  # Sort by List2 values

                # Create a new List1 based on the sorted order of List2
                all_files_sorted = [item[1] for item in sorted_combined]
                
                
                
                # Process the images and create a DataFrame to store the results
                results_df = process_images(all_files_sorted)
                df = results_df.set_index('Image No.')
                
                
                
                # GRAPHS CODE
                files = [f.name for f in all_files_sorted]
                filenames = [f.split(".")[0] for f in files]
                
                
                # Group by 'Image' and calculate the mean for each group
                results_df_avg = results_df.groupby('Image No.').mean()
                
                
                # Reset the index (optional, but makes the DataFrame cleaner)
                results_df_avg = results_df_avg.reset_index()
                
                # results_df_avg = pd.concat([pd.DataFrame([new_row]), results_df_avg], ignore_index=True)
                
                def time_difference_in_seconds(t1, t2):
                    year_diff = (int(t2[:4]) - int(t1[:4])) * 365 * 24 * 3600  # Year difference
                    month_diff = (int(t2[4:6]) - int(t1[4:6])) * 30 * 24 * 3600  # Month difference (approx 30 days)
                    day_diff = (int(t2[6:8]) - int(t1[6:8])) * 24 * 3600  # Day difference
                    hour_diff = (int(t2[8:10]) - int(t1[8:10])) * 3600  # Hour difference
                    minute_diff = (int(t2[10:12]) - int(t1[10:12])) * 60  # Minute difference
                    second_diff = int(t2[12:14]) - int(t1[12:14])  # Second difference
                    
                    return year_diff + month_diff + day_diff + hour_diff + minute_diff + second_diff
                
                # Calculate time differences in hours with reference to the first image
                
                reference = filenames[0]
                time_differences = [time_difference_in_seconds(reference, f) / 3600 for f in filenames]

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
                
                
                
                
                
                
                
                # Display the results in a table
                st.markdown("<h2 style='text-align: center;'>Results</h2>", unsafe_allow_html=True)
                st.dataframe(df)
                
                # Animate the images side by side
                st.markdown("<h3 style='text-align: center;'>Image Processing Steps</h3>", unsafe_allow_html=True)
                
                # Create placeholders for extracted and processed images
                dynamic_placeholder = st.empty()
                
                caption1 = ['Orginial Image', 'Greyscale Image', 'Triangular Thresholding', 'Morphological Opening', 'Morphological Closing', 'Extracted Regions']
                caption2 = [f'ROI {i+1}' for i in range(0, len(r_lab_values))]
                caption3 = ['L Channel', 'a Channel', 'b Channel']
                
                captions = caption1 + caption2 + caption3
                
                st.markdown("<h2 style='text-align: center;'>Graphs</h2>", unsafe_allow_html=True)

                if selected_params.get("Browning Index"):
                    graph1_placeholder = st.empty()
                    graph1_placeholder.image('output/bi_plot.png', caption='Browning Index Plot', use_column_width=True)
                if selected_params.get("L* a* b*"):
                    graph2_placeholder = st.empty()
                    graph2_placeholder.image('output/lab_plot.png', caption='L*a*b* Plot' ,use_column_width=True)
                
                
                
                image_folder = 'output'  # Update this path to your folder containing the images


                zip_data = zip_images_and_dataframe(image_folder, df)

                st.download_button(
                    label="Download Data (ZIP)",
                    data=zip_data,
                    file_name="Dynamic Data.zip",
                    mime="application/zip"
                )
                
                while (True):
                    # Display extracted images
                    count = 0
                    for image_path in r_images_to_display:
                        dynamic_placeholder.image(image_path, caption=captions[count], use_column_width=True)
                        count+=1# Display the current extracted image
                        time.sleep(3)  # Wait for 1 second before displaying the next image
