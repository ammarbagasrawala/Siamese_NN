import tensorflow as tf
import onnxruntime as ort
from PIL import Image
import numpy as np

def preprocess_pil(image):
  if not isinstance(image, Image.Image):
    raise ValueError(f"Input must be a PIL Image But got {type(image)}")

  img = np.array(image)
  img = tf.image.resize(img, (105, 105))
  img = img / 255.0
  return img

# def preprocess_twins_pil(input_img, given_img):
#   return (preprocess_pil(input_img), preprocess_pil(given_img))

# def load_and_run_onnx_model(model_path, input_imgs, validation_img):
#   onnx_session = ort.InferenceSession(model_path,providers=['CPUExecutionProvider'])

#   input_imgs = np.stack([preprocess_pil(img) for img in input_imgs])
#   validation_img = preprocess_pil(validation_img)

#   input_imgs = tf.convert_to_tensor(input_imgs, dtype=tf.float32)
#   validation_imgs = tf.convert_to_tensor(np.tile(validation_img, (len(input_imgs), 1, 1, 1)), dtype=tf.float32)

#   input_names = [input.name for input in onnx_session.get_inputs()]
#   output_names = [output.name for output in onnx_session.get_outputs()]

#   input_data = {input_names[0]: input_imgs.numpy(), input_names[1]: validation_imgs.numpy()}

#   output = onnx_session.run(output_names, input_data)

#   return output


def load_and_run_onnx_model(model_path, input_imgs, input_filenames, validation_img):
    onnx_session = ort.InferenceSession(model_path)

    input_imgs = np.stack([preprocess_pil(img) for img in input_imgs])
    validation_img = preprocess_pil(validation_img)

    input_imgs = tf.convert_to_tensor(input_imgs, dtype=tf.float32)
    validation_imgs = tf.convert_to_tensor(np.tile(validation_img, (len(input_imgs), 1, 1, 1)), dtype=tf.float32)

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    input_data = {input_names[0]: input_imgs.numpy(), input_names[1]: validation_imgs.numpy()}

    output = onnx_session.run(output_names, input_data)
    
    # Associate similarity scores with filenames
    similarity_scores = output[0]
    filename_to_similarity = dict(zip(input_filenames, similarity_scores))

    # Find the filename with the maximum similarity score
    most_similar_filename = max(filename_to_similarity, key=filename_to_similarity.get)
    max_similarity_score = filename_to_similarity[most_similar_filename]

    return most_similar_filename ,max_similarity_score

