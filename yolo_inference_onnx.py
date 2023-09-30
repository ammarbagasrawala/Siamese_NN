import cv2
from PIL import Image
import numpy as np
import os



def drawbbox(imgin, boxes, confidences, classes_ids, classes):
    crop = []
    box = []
    img = np.copy(imgin)
    # print(classes_ids)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        box.append(boxes[i])
        classlist = classes
        label = classlist[classes_ids[i]]
        conf = confidences[i]
        text = label + " : " + "{:.2f}".format(conf)
        crop.append(img[y1:y2, x1:x2, :])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        
    return img


def get_bboxes(image_obj, path_to_onnx, prediction_mode="NA"): #fname,imgdir):image_obj,
    
    YOLOWEIGHTS=path_to_onnx

    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(YOLOWEIGHTS)
    if prediction_mode == "logo":
            CLASSES = ['logo']
    elif prediction_mode == "table":
        CLASSES = ['table']
    else:
        raise ValueError("Invalid prediction_mode. It should be 'logo' or 'metatable'.")

    img = image_obj
    # img = Image.open("C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\ammar_upscaled.jpg")#os.path.join(imgdir,"{}.png".format(fname)))
    img = image_obj
    original_image = np.array(img)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=False)
    model.setInput(blob)
    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.20:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    detections = []
    bboxes = []
    confidences = []
    classes_ids = []
    indices = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        bboxes.append([round(box[0] * scale), 
                       round(box[1] * scale),
                       round((box[0] + box[2]) * scale), 
                       round((box[1] + box[3]) * scale)])
        confidences.append(scores[index])
        classes_ids.append(class_ids[index])
        indices.append(index)
        
    # img = drawbbox(img, bboxes, confidences, classes_ids, CLASSES) 
    # img = Image.fromarray(img)
        
    return bboxes,img

# get_bboxes("C:\\Users\\ammar\\Documents\\sequus internship\\modularise_code\\Copy of yolov8n.onnx")
