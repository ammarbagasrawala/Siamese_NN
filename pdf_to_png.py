import fitz  # PyMuPDF
import os
from PIL import Image
def pdf_to_png_fitz(pdf_file_path, desired_width=4967, desired_height=3509):#output_folder_path,
    # Create the output folder if it doesn't exist
    # os.makedirs(output_folder_path, exist_ok=True)

    pil_img_list=[]

    # Open the PDF file
    pdf_document = fitz.open(pdf_file_path)
    # print(pdf_document)

    for i, page in enumerate(pdf_document):
        pdf_width = page.rect.width  # Width of the PDF page in points
        pdf_height = page.rect.height  # Height of the PDF page in points
        # print(page)
        
        # Calculate the DPI needed to obtain the desired image size
        dpi_x = (desired_width / pdf_width) * 72
        dpi_y = (desired_height / pdf_height) * 72

        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi_x / 72, dpi_y / 72))
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        # print(image)

        # pil_img_list.append(pil_img)
        pil_img_list.append(image)

        # image.save(os.path.join(output_folder_path, f"{os.path.basename(pdf_file_path)}_page_{i + 1}.png"))
        # # # print()


    pdf_document.close()
    return pil_img_list

# # Example usage:
# pdf_file_path = "C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\Ammar's Resume 30th August 2023 R.pdf"
# output_folder_path = 'C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code'
# # desired_width = 4967  # Desired output image width in pixels
# # desired_height = 3509  # Desired output image height in pixels

# # pdf_to_png_fitz(pdf_file_path, output_folder_path)
