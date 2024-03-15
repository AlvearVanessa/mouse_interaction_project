"""
AI Virtual Mouse
Mon, 07/03/24 13:07H CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/
https://www.assemblyai.com/blog/mediapipe-for-dummies/

This code made the Tesselation over the face in a webcam in real-time
"""

import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
from PIL import Image
from IPython import display
from IPython.display import Video
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hand_mesh = mp.solutions.hands # .hands_connections


# capture the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(3))
height = int(cap.get(4))
size = (width, height)

pTime = 0
cTime = 0

while True:
    # Read the frames from webcam
    success, img = cap.read()

    # Define image filename and drawing specifications
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Create a face mesh object
    with mp_hand_mesh.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hand_mesh:

        # Read image file with cv2 and process with face_mesh
        # image = cv2.imread(file)
        results = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    # Define boolean corresponding to whether or not a face was detected in the image
    hand_found = bool(results.multi_hand_landmarks)
    # print(results.multi_hand_landmarks[0])

    if hand_found:
        # Create a copy of the image
        # annotated_image = img.copy()

        # Draw landmarks on face
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=results.multi_hand_landmarks[0],
            connections=mp_hand_mesh.HAND_CONNECTIONS,
        landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        # Save image
        # cv2.VideoWriter("face_tesselation_only.mp4", vid_cod, 20.0, size)
        # cv2.imwrite('face_tesselation_only.png', annotated_image)
        # cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)

    # Open image
    # img = Image.open('face_tesselation_only.png')
    # display(img)

    black = np.zeros((350, 500, 3), dtype = np.uint8)

    if hand_found:
        # Create a copy of the image
        annotated_image = black

    #print(annotated_image)

        # Draw landmarks on face
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.multi_hand_landmarks[0],
            connections=mp_hand_mesh.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        # Save image
        # cv2.imwrite('face_tesselation_only.png', annotated_image)
        # cv2.VideoWriter("face_tesselation_only.mp4", vid_cod, 20.0, size)

    # Open image
    # img = Image.open('face_tesselation_only.png')
    # display(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    text_to_print = str(int(fps)) # to reverse [::-1]
    cv2.putText(annotated_image, text_to_print, (100, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hand Skeleton Detection - Black Background', cv2.flip(annotated_image, 1))
    # cv2.imshow("Image", annotated_image)
    cv2.waitKey(1)


cap.release()