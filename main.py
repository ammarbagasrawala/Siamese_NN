import argparse
import pprint
from compile_modules import run_logo_module

def main(pdf_directory_path, table_weights, logo_weights, onnx_model_path, input_image_path):
    result_pdf_data_list = run_logo_module(pdf_directory_path, table_weights, logo_weights, onnx_model_path, input_image_path)
    
    # Print the pdf_data_list with pretty formatting
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result_pdf_data_list)
    
    # Write the pdf_data_list to a text file
    with open('pdf_data_list.txt', 'w') as file:
        pp = pprint.PrettyPrinter(stream=file, indent=5)
        pp.pprint(result_pdf_data_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs and perform logo recognition.")
    parser.add_argument("--pdf_dir", required=True, help="Path to the directory containing PDF files.")
    parser.add_argument("--table_weights", required=True, help="Path to table detection model weights (onnx format).")
    parser.add_argument("--logo_weights", required=True, help="Path to logo detection model weights (onnx format).")
    parser.add_argument("--onnx_model", required=True, help="Path to the Siamese logo recognition model (onnx format).")
    parser.add_argument("--input_imgs", required=True, help="Path to the directory containing Input images (templates).")
    
    args = parser.parse_args()
    main(args.pdf_dir, args.table_weights, args.logo_weights, args.onnx_model, args.input_imgs)
