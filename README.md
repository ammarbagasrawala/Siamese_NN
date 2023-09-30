# Siamese_NN
Built a Siamese Neural Networks for One-shot Logo Recognition using the research paper Siamese Neural Networks for One-shot Image Recognition
(Gregory Koch GKOCH@CS.TORONTO.EDU
Richard Zemel ZEMEL@CS.TORONTO.EDU
Ruslan Salakhutdinov RSALAKHU@CS.TORONTO.EDU)

Siamese Neural Network is not a classfier its a comparator. 
Input and a Validation image in fed in during inferencing and either 1 or 0 is the output.

# How to use
for inferencing make a template_inputs_directory containig logo crops and filename should be the logo name (these are the images with which the incoming image will be compared to)
specify the path in compile modules

Run _python main.py --pdf_dir "path_to_pdf_dir" --table_weights "path_to_table_recognition_weights" --logo_weights "path_to_logo_recognition_weights" --onnx_model "path_to_siamese_weights" --input_imgs "path_to_input_images"_

# Note 
all the model weights should be in ONNX format only
