import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

global params

params = cv2.SimpleBlobDetector_Params()
 
params.minThreshold = 10;
params.maxThreshold = 200;
params.filterByArea = True
params.filterByCircularity = False
params.minArea = 500
params.filterByConvexity = True;
params.minConvexity = 0.40;

detector = cv2.SimpleBlobDetector_create(params)

threshold = 65

global first_frame

global rms_y, counter 
rms_y = 0
counter = 0

def update_threshold(val):
    global threshold
    threshold = int(val)
    plt.title(f'Threshold Value: {threshold}')
    plt.draw()

# def update_min_area(keypoints):
#     global first_frame
#     global green
#     if len(keypoints) == 1 and first_frame is True:
#         # params.minArea = int(keypoints[0].size * 0.95)
#         green = int(10000)
#         print(f"Updated minArea: {params.minArea}")
#         first_frame = False

def detect_eyes (img):
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

    left_eye = None
    right_eye = None

    for (x,y,w,h) in faces:
        gray_face = gray_picture[y:y+h, x:x+w]
        face = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face)

        for (ex,ey,ew,eh) in eyes:
            if ey > y * 0.5:
                if ex < w * 0.7:
                    left_eye = gray_face[y:ey + eh , x:ex + ew]
                    print(left_eye)
                # if ex > w * 0.5:
                    # right_eye = gray_face[y:ey + eh , x:ex + ew]
    return left_eye,right_eye

def cut_eyebrow_region (img):
    height, width = img.shape[:2]
    non_eyebrow_region = int(height/4)
    img = img[non_eyebrow_region: height, 0: width]

    return img

def blob_process(img):
    if img is None or img.size == 0:
        return []

    global threshold
    _, eye_roi = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    eye_roi = cv2.erode(eye_roi, None, iterations=4)
    eye_roi = cv2.dilate(eye_roi, None, iterations=9)
    eye_roi = cv2.medianBlur(eye_roi, 5) 

    # eye_roi = cv2.resize(eye_roi, None, fx=2.5, fy=2.5)
    # cv2.imshow("Original ROI", eye_roi)
    
    keypoints = detector.detect(eye_roi)

    # print(keypoints)
   

    return keypoints, eye_roi

def main():
    global first_frame
    first_frame = True;

    x_values = []
    y_values = []

    def animate(x,y):
        x_values.append(x)
        y_values.append(y)
        plt.cla()
        plt.plot(x_values, y_values)
        plt.axhline(y=402, color='r', linestyle='-')
        print(x,height-y)
        print(sum(y_values) / len(y_values) if y_values else 0)

    video_capture = cv2.VideoCapture('circle.mp4')
   
    plt.ion()

   

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        height, width, _ = frame.shape
        frame = frame[:, width // 2:]
        # frame = frame[:, :width//2]

        if frame is not None:
           
            keypoints, eye_roi = blob_process(frame)

            key_eye_roi = cv2.drawKeypoints(eye_roi, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            key_frame = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2BGR)
            window = cv2.hconcat([key_eye_roi, key_frame])
            cv2.imshow("Result", window)

            for keypoint in keypoints:
                x, y = map(int, keypoint.pt)

                #RMS
                global rms_y, counter
                rms_y = rms_y + ((height - y)-402) ** 2
                counter += 1

                print(f"RMS: {np.sqrt(rms_y/counter)}")

                ani = FuncAnimation(plt.gcf(), animate(x,y), 1000)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    plt.ioff()
    plt.show()

    video_capture.release()
    cv2.destroyAllWindows()

    print(f"RMS: {np.sqrt(rms_y/counter)}")

if __name__=="__main__": 
    main() 

