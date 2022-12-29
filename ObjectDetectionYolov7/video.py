import cv2
import sys
sys.path.append('./yolov7')
from load_model import load_model
from tqdm import tqdm
from smooth_results import SmoothedVideo
import numpy as np


def video(video_path, show_video=True, save_video=False, video_name=None):
    """"
        Processes a video file specified by the video_path parameter. 
        
    Parameters:
         video_path: a path to the video that we want to detect.
         show_video: a boolean flag that determines whether the processed video should be displayed. 
                     The default value is True.
         save_video: a boolean flag that determines whether the processed video should be saved. 
                     The default value is False.
         video_name: a string that specifies the name of the saved video file. 
                    If not provided, the default value is None

    Returns:
         None
    """""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    length = int(cv2.VideoCapture.get(cap, property_id))
    model = load_model('./yolov7/cfg/training/yolov7-tiny-exp1.yaml',
                       './yolov7/runs/train/exp/weights/best.pt')
    sv = SmoothedVideo(model, smooth_thres=0.95)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    if save_video:
        # Below VideoWriter object will create
        # a frame of above defined The output
        # is stored in 'filename.avi' file.
        result = cv2.VideoWriter(f'{video_name}.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, size)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    for t in tqdm(range(length)):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # Convert to RGB
            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame
            img, __ = sv.make_smooth(im_rgb)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            if save_video:
                result.write(img)

            if show_video:

                cv2.imshow('Frame', img)
                cv2.waitKey(1)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    # When everything done, release
    # the video capture and video
    # write objects
    cap.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()
