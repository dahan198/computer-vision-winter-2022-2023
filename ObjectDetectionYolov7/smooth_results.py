import torch
import numpy as np
from yolov7.utils.general import non_max_suppression
from yolov7.utils.plots import plot_images, output_to_target
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
        if len(target) == 0:
            return target
        target[:, 0] = 0
        return target

    def make_smooth(self, image):
        """"
            Perform object detection on the given image and applies smoothing to the resulting predictions.

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
        if len(target) <= 1:
            target = self.predictions[-1]
        else:
            target = self.remain_left_right_labels(target)  # removes duplicate boxes
            target = self.fix_uncertain_labels(target)  # fixes uncertain labels

        # Add the predictions for this image to the list of all predictions
        self.predictions.append(target)

        # Create an image with the predictions drawn on it
        image_result = Image.fromarray(plot_images(img, targets=target, names=self.names))

        return image_result, target[:, 1]

    def remain_left_right_labels(self, target):
        """"
            Remain only two labels, one for the left hand and the another for the right hand.

        Args:
           target (np.array): a 2D NumPy array containing multiple bounding boxes information and their labels.
        
        Returns:
           numpy array: a 2D NumPy array containing two bounding boxes information and their labels.
        """""

        # Get the number of bounding boxes in the target array
        boxes_num = len(target)
        # Get a boolean array indicating whether the label is odd or even
        odd = (target[:, 1] % 2).astype(bool)

        # If there are only two boxes, check if they are both odd or both even
        # If they are, return the boxes. If not, return the box with the highest confidence
        if boxes_num == 2:
            if odd[0] != odd[1] and np.linalg.norm(target[0][2] - target[1][2]) > 50:
                return target
            else:
                max_conf = np.argmax(target[:, 6])
                return self.add_second_label(target[max_conf])

        # Get the indices of the odd and even labels
        indices = np.array(range(boxes_num))
        left = indices[odd]
        left_confidences = target[odd, 6]
        right = indices[~odd]
        right_confidences = target[~odd, 6]

        # If there are no odd or even labels, return the box with the highest confidence
        if len(right) == 0 or len(left) == 0:
            max_conf = np.argmax(target[:, 6])
            return self.add_second_label(target[max_conf])

        # Sort the right and left labels by confidence in descending order
        best_right = list(reversed(np.argsort(right_confidences)))
        best_left = list(reversed(np.argsort(left_confidences)))

        # Calculate the distance between each pair of right and left labels
        # If the distance is greater than 50, return the pair
        distances = []
        indices_by_distance = []
        for right_idx in best_right:
            for left_idx in best_left:
                distances.append(np.linalg.norm(target[right[right_idx]][2] - target[left[left_idx]][2]))
                indices_by_distance.append([right_idx, left_idx])
                if distances[-1] > 50:
                    return target[[right[right_idx], left[left_idx]]]

        # If no pairs are found with a distance greater than 50, return the box with the highest confidence
        max_conf = np.argmax(target[:, 6])
        return self.add_second_label(target[max_conf])

    def add_second_label(self, target):
        """"
            Add a second label to the target array. 
            The label is selected from the predictions array using the distance from the target label.
            The label with the maximum distance is chosen.
            
        Parameters:
            target (numpy array): array containing a single label with 7 elements  
                                  [label, x, y, w, h, confidence, class_id]

        Returns:
            numpy array: array containing 2 labels with 7 elements each
        """""

        # Calculate the distance between the target label and all the labels in the predictions array
        distance = np.linalg.norm(self.predictions[-1][:, 2:3] - target[2], axis=1)

        # Get the index of the label with the maximum distance
        second_index = np.argmax(distance)

        # Modify the label in the target array to be the label with the maximum distance
        target[1] = self.predictions[-1][1 - second_index][1]

        # Return the target array and the label with the maximum distance as a numpy array
        return np.vstack([target, self.predictions[-1][second_index]])

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



