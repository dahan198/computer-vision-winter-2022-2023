import torch
import sys
sys.path.append('./yolov7')
from load_model import load_model
from PIL import Image
import numpy as np
from yolov7.utils.general import non_max_suppression
from yolov7.utils.plots import plot_images, output_to_target
import cv2


def predict(image_path, show_predicted_image=True, iou_thres=0.1, conf_thres=0.01):
    """"
        Given an image path and optional parameters, returns a PIL image object with the predicted image.
        The predicted image will be plotted with predicted bounding boxes.
        
    Parameters:
        image_path (str): a string representing the path to the image file.
        show_predicted_image (bool, optional): whether to show the predicted image using the Image.show() method. 
                                               Defaults to False.
        iou_thres (float, optional): the Intersection over Union (IoU) threshold used in the non-max suppression
                                     process. Defaults to 0.1.
        conf_thres (float, optional): the confidence threshold used in the non-max suppression process. 
                                      Defaults to 0.01.

    Returns:
        PIL.Image: a PIL image object with the predicted image.
    """""

    # Choose device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image and preprocess
    img = np.asarray(Image.open(image_path))
    img = torch.tensor(data=img, device=device, dtype=torch.float)
    img /= 255.0
    img = img.unsqueeze(0).permute(0, 3, 1, 2)

    # Make predictions with the model
    model = load_model('./yolov7/cfg/training/yolov7-tiny-exp1.yaml', './yolov7/runs/train/exp/weights/best.pt')
    out, train_out = model(img)
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True)

    # Get class names
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    # Convert output to target format
    target = output_to_target(out)

    # Plot predicted boxes on the image
    image_result = plot_images(img, targets=target, names=names)
    image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

    # Show image with predicted boxes if desired
    if show_predicted_image:
        cv2.imshow('result', image_result)
        cv2.waitKey(0)

    return image_result

