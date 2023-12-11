import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread("stock.jpg")

params = cv2.SimpleBlobDetector_Params()
 
params.minThreshold = 10;
params.maxThreshold = 200;
params.filterByArea = True
params.minArea = 10

detector = cv2.SimpleBlobDetector_create(params)

def detect_face (img):
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
    #Later, add so only 1 can be detected
    for (x,y,w,h) in faces:
       cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 42), 2)

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
                if ex < w * 0.5:
                    left_eye = face[y:ey + eh , x:ex + ew]
                if ex > w * 0.5:
                    right_eye = face[y:ey + eh , x:ex + ew]
        
    return left_eye,right_eye
    
def display_eyes (img):
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

    for (x,y,w,h) in faces:
        gray_face = gray_picture[y:y+h, x:x+w]
        face = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0, 255, 42), 2)

def cut_eyebrow_region (img):
    height, width = img.shape[:2]
    non_eyebrow_region = int(height/4)
    img = img[non_eyebrow_region: height, 0: width]

    return img

def blob_process(img):
    gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, eye_roi = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY)

    eye_roi = cv2.erode(eye_roi, None, iterations=2)
    eye_roi = cv2.dilate(eye_roi, None, iterations=4)
    eye_roi = cv2.medianBlur(eye_roi, 5)
    
    keypoints = detector.detect(eye_roi)
    print(keypoints)

    return keypoints


left_eye, right_eye = detect_eyes(img)
left_eye = cut_eyebrow_region(left_eye)

keypoints = blob_process(left_eye)

left_eye = cv2.drawKeypoints(left_eye, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

left_eye = cv2.resize(left_eye, None, fx=10, fy=10)

cv2.imshow('Left Eye', left_eye)

cv2.waitKey(0)
cv2.destroyAllWindows()



