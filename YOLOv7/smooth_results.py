import torch
import numpy as np
from utils.general import non_max_suppression
from utils.plots import plot_images, output_to_target
from PIL import Image
from collections import Counter


class SmoothedVideo:
    """"
        Processes and smooths video frames using an object detection model.
    """""

    def __init__(self, model, iou_thres=0.0, conf_thres=0.3, smooth_thres=0.95):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        self.predictions = []
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.smooth_thres = smooth_thres

    def detect(self, img):
        """"
            Run object detection on a single image.

        Args:
            img (torch.Tensor): The image to run object detection on.

        Returns:
            torch.Tensor: The object detection results in the format required by utils.plots.output_to_target.
        """""

        # Get model output
        out, train_out = self.model(img)
        out = non_max_suppression(out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, labels=[],
                                  multi_label=True)
        target = output_to_target(out)
        target[:, 0] = 0
        return target

    def make_smooth(self, image):
        """"
            Performs object detection on the given image and applies smoothing to the resulting predictions.

        Parameters:
             image (A PIL image): an array of bounding boxes in the format [x1, y1, x2, y2, class_id, score]

        Returns:
             A PIL image with the predicted bounding boxes overlaid on top
        """""

        # Convert image to a tensor and normalize it
        img = np.asarray(image)
        img = torch.tensor(data=img, device=self.device, dtype=torch.float)
        img /= 255.0
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        # Get predictions for the image
        target = self.detect(img)
        target = target[target[:, 6] > self.conf_thres]  # keeps predictions with a high enough confidence
        target = self.remove_duplicate_boxes(target)  # removes duplicate boxes
        target = target[target[:, 6].argsort(axis=0, kind='mergesort')][-2:, :]  # keeps the top two by confidence
        target = self.fix_uncertain_labels(target)  # fixes uncertain labels

        # Add the predictions for this image to the list of all predictions
        self.predictions.append(target)

        # Create an image with the predictions drawn on it
        image_result = Image.fromarray(plot_images(img, targets=target, names=self.names))

        return image_result

    @staticmethod
    def remove_duplicate_boxes(target):
        """"
            Removes duplicate bounding boxes from the given target array.
            A box is considered a duplicate if its coordinates (x1, y1, x2, y2) are within 2 pixels of another box.
            If two boxes have the same coordinates, the box with the higher score is kept.

        Parameters:
            target (numpy array): an array of bounding boxes in the format [x1, y1, x2, y2, class_id, score]

        Returns:
            target (numpy array): the input array with duplicate boxes removed
        """""

        boxes_num = len(target)  # number of bounding boxes in the target array
        indices = list(range(boxes_num))  # list of indices to keep

        for i in range(boxes_num):
            for j in range(i + 1, boxes_num):
                # If the distance between the coordinates of the two boxes is less than or equal to 2 pixels:
                if np.linalg.norm(target[i][:4] - target[j][:4]) <= 2:
                    # If the score of box i is greater than that of box j:
                    if target[i][5] > target[j][5]:
                        # Remove box j from the list of indices to keep:
                        indices.remove(j)
                    else:
                        # Otherwise, remove box i:
                        indices.remove(i)

        # Return the target array with only the boxes at the indices in the list of indices to keep:
        return target[indices]

    def fix_uncertain_labels(self, target):
        """"
            This function receives the current target (array of detections) and checks if there are any detections
            with low confidence.
                If there are, it checks the last 5 frames to see if any of those detections are consistently present.
                If a detection is consistently present, its label is updated to the most common label among the
                   detections.
                If a detection is not consistently present, it is removed from the target.
                
        Parameters:
            target (numpy array): Array of shape (n_detections, 7) containing the detections for the current frame.

        Returns:
            target (numpy array): Array of shape (n_detections, 7) containing the updated detections for the 
            current frame.
        """""

        if len(self.predictions) > 0:

            # Check the last 5 frames for detections with low confidence
            last_frames = self.predictions[-5:]

            uncertain_cond = target[:, 6] < self.smooth_thres
            uncertain_labels = target[uncertain_cond].copy()
            if len(uncertain_labels) > 0:
                new_uncertain_labels = []
                for i, uncertain_label in enumerate(uncertain_labels):
                    uncertain_label = np.expand_dims(uncertain_label, axis=0)

                    # Check if the detection is consistently present in the last 5 frames and update if it's needed
                    new_uncertain_labels.append(self.get_aggregated_label_and_confidence(uncertain_label, last_frames))

                target[uncertain_cond] = np.vstack(new_uncertain_labels)
        return target

    @staticmethod
    def get_aggregated_label_and_confidence(uncertain_label, last_frames):
        """"
            Given a uncertain label , this function returns a new list of labels with the aggregated label and 
            confidence. The aggregated label is the label that appears the most in the last_frames list, 
            and the aggregated confidence is the mean confidence of all the labels in the last_frames.
    
        Parameters:
        - uncertain_label (numpy array): an array of uncertain labels, where each label is
        a 7-dimensional array containing the label id, bounding box coordinates, and confidence.
        
        Returns:
        - new_uncertain_label (numpy array): an array of aggregated labels.
        """""

        confidence = []  # list of all the fitted confidences
        labels = []  # list of all the fitted labels

        for frame in last_frames:
            if len(frame) > 1:
                distance = np.linalg.norm(uncertain_label[:, 2:6] - np.expand_dims(frame[:, 2:6], 1), axis=2)
                best_indices = np.argsort(distance.flatten())
                best_index = best_indices[0]
                labels.append(frame[best_index, 1])
                confidence.append(frame[best_index, 6])

        if len(labels) > 0:
            c = Counter(labels)
            most_common = c.most_common(1)

            # Add new labels with the aggregated label and confidence
            uncertain_label[0, 1] = most_common[0][0]
            uncertain_label[0, 6] = np.average(confidence)
        return uncertain_label



