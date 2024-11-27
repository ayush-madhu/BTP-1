import zipfile
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_zip(images, graphs, dataframe):
    
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Ensure the folder structure exists inside the ZIP file
        zf.writestr("Images/", "")  # Create the Images folder
        zf.writestr("Graphs/", "")  # Create the Graphs folder
        zf.writestr("Table/", "")   # Create the Table folder

        # Add images to the "Images/" folder
        for idx, img in enumerate(images):
            if isinstance(img, np.ndarray):  # If img is a NumPy array, convert it to PIL Image
                img = Image.fromarray(img)
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            zf.writestr(f"Images/image_{idx + 1}.png", img_buffer.getvalue())

        # Add graphs to the "Graphs/" folder
        for idx, graph in enumerate(graphs):
            # If graph is a Matplotlib figure, save it as a PNG image
            if isinstance(graph, plt.Figure):
                graph_buffer = BytesIO()
                graph.savefig(graph_buffer, format="PNG")
                graph_buffer.seek(0)
                zf.writestr(f"Graphs/graph_{idx + 1}.png", graph_buffer.getvalue())
            elif isinstance(graph, np.ndarray):  # If graph is a NumPy array, convert it to PIL Image
                graph_image = Image.fromarray(graph)
                graph_buffer = BytesIO()
                graph_image.save(graph_buffer, format="PNG")
                graph_buffer.seek(0)
                zf.writestr(f"Graphs/graph_{idx + 1}.png", graph_buffer.getvalue())

        # Add the DataFrame as an Excel file in the "Table/" folder
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False)
        excel_buffer.seek(0)
        zf.writestr("Table/results_data.xlsx", excel_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer
