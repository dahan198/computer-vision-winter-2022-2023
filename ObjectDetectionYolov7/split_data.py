import shutil
from tqdm import tqdm
import os


""""
    Arranges data in yolo-v7 format
"""""


# Set the path to the source and destination directories
images_path = 'HW1_dataset/images/'
dataset_types = ['train', 'valid', 'test']
src_dir = "./HW1_dataset/"

# Iterate through each dataset type
for dataset_type in dataset_types:

    # Set the destination directory for the current dataset type
    dst_dir = "./yolov7/arranged_dataset/" + dataset_type + '/'

    # Create the destination directory if it does not already exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir + 'images/')
        os.makedirs(dst_dir + 'labels/')

    # Open the file containing the list of images for the current dataset type
    with open(src_dir + dataset_type + '.txt', 'r') as f:
        lines = f.readlines()

    # Iterate through each image in the list
    for line in tqdm(lines):

        # Remove the newline character from the image name
        line = line.split('\n')[0]

        # Copy the image and its label to the destination directory
        shutil.copy(src_dir + 'images/' + line, dst_dir + 'images/')
        shutil.copy(src_dir + 'bboxes_labels/' + line.split('.')[0] + '.txt', dst_dir + 'labels/')

