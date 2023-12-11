import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

params = cv2.SimpleBlobDetector_Params()
 
params.minThreshold = 10;
params.maxThreshold = 200;
params.filterByArea = True
params.filterByCircularity = False
params.minArea = 100
params.filterByConvexity = True;
params.minConvexity = 0.40;

detector = cv2.SimpleBlobDetector_create(params)

threshold = 65

def update_threshold(val):
    global threshold
    threshold = int(val)
    plt.title(f'Threshold Value: {threshold}')
    plt.draw()

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
    print(keypoints)
   

    return keypoints, eye_roi

def main():
    video_capture = cv2.VideoCapture('circle.mp4')

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    thresh_slider_ax = plt.axes(facecolor='lightgoldenrodyellow')

    thresh_slider = Slider(
    ax=thresh_slider_ax,
    label='Threshold',
    valmin=10,
    valmax=245,
    valinit=threshold,
    )
    thresh_slider.on_changed(update_threshold)

    while True:
        ret, frame = video_capture.read()

        

        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning when it ends
            continue

        height, width, _ = frame.shape
        frame = frame[:, :width//2]

        if frame is not None:
            keypoints, eye_roi = blob_process(frame)
            key_eye_roi = cv2.drawKeypoints(eye_roi, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            key_frame = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2BGR)
            window = cv2.hconcat([key_eye_roi, key_frame])
            cv2.imshow("Result", window)
            
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        plt.pause(0.01)

    plt.show()

    video_capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__": 
    main() 

