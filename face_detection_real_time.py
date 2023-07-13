import cv2
import matplotlib.pyplot as plt

# we are using a trained model
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# we will use real time default camera - 0 means the default camera
video_capture = cv2.VideoCapture(0)

# setting the width and height of the video
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    # return the next video frame (the img is important)
    ret, img = video_capture.read()

    # transform into grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use face detection cascade classifier algorithm
    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    for (X, Y, width, height) in detected_faces:
        cv2.rectangle(img, (X, Y), (X + width, Y + height), (0, 0, 255), 10)

    # title of video window
    cv2.imshow('Real-Time Fcae Detection', img)

    # we wait for a key to be presented - press 'ESC' to quit
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# destroy and release the camera etc..
video_capture.release()
cv2.destroyAllWindows()