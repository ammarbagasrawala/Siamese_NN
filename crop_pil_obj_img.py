from PIL import Image

def crop_images(image_obj, bboxes):
    cropped_images = []

    for bbox in bboxes:
        left, top, right, bottom = bbox
        cropped_image = image_obj.crop((left, top, right, bottom))
        cropped_images.append(cropped_image)

    return cropped_images