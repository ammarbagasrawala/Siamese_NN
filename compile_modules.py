from pdf_to_png import pdf_to_png_fitz
from yolo_inference_onnx import get_bboxes
from crop_pil_obj_img import crop_images
from PIL import Image
from siamese_logo_recognition import load_and_run_onnx_model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def has_pdf_files(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):
                return True
    return False

def convert_pdfs_in_directory(directory_path):
    pdf_data_list = []

    if not has_pdf_files(directory_path):
        print(f"No PDF files found in the directory '{directory_path}' or its subdirectories.")
        return pdf_data_list

    for root, _, files in os.walk(directory_path):
        # print("######################")
        # print(root)
        if not os.listdir(root):
            print(f"No files found in the directory '{root}'.")
            continue
        for file in files:
            if file.endswith(".pdf"):
                # print(file)
                pdf_file_path = os.path.join(root, file)
                pil_img_list = pdf_to_png_fitz(pdf_file_path)

                # Store folder name, file name, and PIL images in a dictionary
                pdf_data = {
                    'folder_name': os.path.basename(root),
                    'file_name': file,
                    'images': pil_img_list
                }

                pdf_data_list.append(pdf_data)

    return pdf_data_list

# print("hello")


# pdf_directory_path='pdf_dir'
# pdf_data_list=convert_pdfs_in_directory(pdf_directory_path)

# for pdf_data in pdf_data_list:
#     # Get the images and weights
#     pdf_images = pdf_data['images']
#     table_weights = "C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\metatable_best.onnx"
#     logo_weights="C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\logo_best.onnx"
#     # Initialize a list to store table_bboxes for each image in the PDF
#     # metatable_bboxes = []

#     # Process each image in the PDF
#     for image in pdf_images:
#         # Get table_bboxes and table_img for the current image
#         table_bboxes, table_img = get_bboxes(image, table_weights, prediction_mode='table')
#         table_crops=crop_images(table_img,table_bboxes)
#         # logo_bboxes=[]
#         for metatable in table_crops:
#             logo_bboxes,logo_img=get_bboxes(metatable,logo_weights,prediction_mode='logo')
#             # logo_bboxes.append(logo_bbox)
#             pdf_data['logo_bboxes'] = logo_bboxes
#             logo_crops=crop_images(logo_img,logo_bboxes)
#             pdf_data['logo_crops']=logo_crops
#             # logo_img.show()
#         # table_img.show()
        
#         # Store table_bboxes for the current image in metatable_bboxes
#         # metatable_bboxes.append(table_bboxes)

#         # Store metatable_bboxes in the PDF data
#         pdf_data['metatable_bboxes'] = table_bboxes

import pprint

# Print the pdf_data_list with pretty formatting
pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(pdf_data_list)


# # pdf_data_list[0]['logo_crops'][0].show()

# onnx_model_path = 'Siamese_Weights_for_logo_Iteration_1_v2.onnx'
# for pdf_data in pdf_data_list:
#     # Load one validation image and multiple input images
    
#     validation_imgs = pdf_data['logo_crops']
#     input_imgs_pil = [Image.open(os.path.join('input_imgs', filename)) for filename in os.listdir('input_imgs')]
#     # List of input filenames corresponding to input images
#     input_filenames = os.listdir('input_imgs')
#     # pp.pprint(input_filenames)
#     # pp.pprint(input_imgs_pil)
#     maximum_similarity=0
#     siamese_list=[]
#     print('############',pdf_data['file_name'],'###############')
#     for validation_img_pil in validation_imgs:
#         # Call the inference function with one validation image and multiple input images
#         most_similar_filename , max_similarity_score = load_and_run_onnx_model(onnx_model_path, input_imgs_pil, input_filenames, validation_img_pil)
#         siamese_list.append([most_similar_filename,max_similarity_score[0]])
#         # 'output' contains similarity scores
#         # print(pdf_data['file_name'] , max_similarity_score , most_similar_filename)
#         # print("max_similarity_score:", max_similarity_score)

#         # # 'most_similar_filename' contains the filename of the most similar image
#         # print("Most Similar Filename:", most_similar_filename)
#     # pp.pprint(siamese_list)
#     max_score_list = max(siamese_list, key=lambda x: x[1])
#     consultant_name = max_score_list[0]
#     pdf_data['logo_name'] = consultant_name[:-4]
#     pdf_data['logo_similarity_score'] = max_score_list[1]
#     print(consultant_name[:-4])

#         # maximum_similarity=max(maximum_similarity,most_similar_filename)


# pp.pprint(pdf_data_list)




def process_pdf_data(pdf_data_list, table_weights, logo_weights, onnx_model_path, input_image_path):
    for pdf_data in pdf_data_list:
        pdf_images = pdf_data['images']

        for image in pdf_images:
            table_bboxes, table_img = get_bboxes(image, table_weights, prediction_mode='table')
            table_crops = crop_images(table_img, table_bboxes)

            for metatable in table_crops:
                logo_bboxes, logo_img = get_bboxes(metatable, logo_weights, prediction_mode='logo')
                pdf_data['logo_bboxes'] = logo_bboxes
                logo_crops = crop_images(logo_img, logo_bboxes)
                pdf_data['logo_crops'] = logo_crops

    for pdf_data in pdf_data_list:
        validation_imgs = pdf_data['logo_crops']
        input_imgs_pil = [Image.open(os.path.join(input_image_path, filename)) for filename in os.listdir(input_image_path)]
        input_filenames = os.listdir(input_image_path)

        maximum_similarity = 0
        siamese_list = []

        for validation_img_pil in validation_imgs:
            most_similar_filename, max_similarity_score = load_and_run_onnx_model(
                onnx_model_path, input_imgs_pil, input_filenames, validation_img_pil)
            siamese_list.append([most_similar_filename, max_similarity_score[0]])

        max_score_list = max(siamese_list, key=lambda x: x[1])
        consultant_name = max_score_list[0]
        pdf_data['logo_name'] = consultant_name[:-4]
        pdf_data['logo_similarity_score'] = max_score_list[1]
        print(consultant_name[:-4])

def run_logo_module(pdf_directory_path, table_weights, logo_weights, onnx_model_path, input_image_path):
    pdf_data_list = convert_pdfs_in_directory(pdf_directory_path)
    process_pdf_data(pdf_data_list, table_weights, logo_weights, onnx_model_path, input_image_path)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(pdf_data_list)
    
    # Return the pdf_data_list dictionary
    return pdf_data_list

# if __name__ == "__main__":
#     pdf_directory_path = 'pdf_dir'
#     table_weights = "C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\metatable_best.onnx"
#     logo_weights = "C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\logo_best.onnx"
#     onnx_model_path = 'Siamese_Weights_for_logo_Iteration_1_v2.onnx'
#     input_image_path = 'input_imgs'

#     result_pdf_data_list = run_logo_module(pdf_directory_path, table_weights, logo_weights, onnx_model_path, input_image_path)

#     pp.pprint(result_pdf_data_list)























# file_path = 'test_dict_info.txt'

# # Open the file in write mode and write the dictionary as a string
# with open(file_path, 'w') as file:
#     for pdf_data in pdf_data_list:
#         file.write("File Name: {}\n".format(pdf_data['file_name']))
#         file.write("Folder Name: {}\n".format(pdf_data['folder_name']))
#         file.write("Logo Name: {}\n".format(pdf_data.get('logo_name', 'N/A')))  # Add 'logo_name' if available
#         file.write("Meta Table Bboxes: {}\n".format(pdf_data.get('metatable_bboxes', 'N/A')))
#         file.write("Logo Bboxes: {}\n".format(pdf_data.get('logo_bboxes', 'N/A')))  # Add 'metatable_bboxes' if available
#         file.write("Logo Similarity Score: {}\n".format(pdf_data.get('logo_similarity_score', 'N/A')))
#         # Add more fields as needed
#         file.write("\n")
