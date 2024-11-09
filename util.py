import os
import zipfile
import pandas as pd
from io import BytesIO


def zip_images_and_dataframe(image_folder, dataframe):
    zip_buffer = BytesIO()  # Create a BytesIO buffer to hold the zip file data
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        # Add images from the folder to the zip file with folder structure
        for root, _, files in os.walk(image_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip with relative path to maintain folder structure
                relative_path = os.path.relpath(file_path, image_folder)
                with open(file_path, "rb") as img_file:
                    zf.writestr(relative_path, img_file.read())

        # Convert DataFrame to Excel and add it to the zip file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False)
        excel_buffer.seek(0)  # Move the cursor to the start of the buffer
        zf.writestr("table_data.xlsx", excel_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer
