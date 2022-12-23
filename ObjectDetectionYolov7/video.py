import cv2
from load_model import model
from tqdm import tqdm
from smooth_results import SmoothedVideo

cap = cv2.VideoCapture('./videos/P022_balloon1.wmv')
property_id = int(cv2.CAP_PROP_FRAME_COUNT)
length = int(cv2.VideoCapture.get(cap, property_id))

sv = SmoothedVideo(model)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
i=0
# Read until video is completed
# while (cap.isOpened()):
for t in tqdm(range(length)):
    i+=1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        # frame = predict(frame)
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sv.make_smooth(im_rgb).save('a/' + str(i) + '.jpg')
        # cv2.imshow('Frame', np.asarray(frame))
        #
        # # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
