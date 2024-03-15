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
import time
import math
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


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hand_mesh = mp.solutions.hands # .hands_connectionsmp_pose = mp.solutions.pose



# capture the webcam
#cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
#vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
#width = int(cap.get(3))
#height = int(cap.get(4))
#size = (width, height)




class handDetector():
    def __init__(self, mode=True, maxHands=1, modelC=1, detectionCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC,
                                        self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # Create a face mesh object
        with self.mpHands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hand_mesh:
            results = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Define boolean corresponding to whether or not a face was detected in the image
        hand_found = bool(results.multi_hand_landmarks)
        if hand_found:
            # Draw landmarks on face
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=self.results.multi_hand_landmarks[0],
                connections=self.mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        black = np.zeros((350, 500, 3), dtype=np.uint8)
        if hand_found:
            # Create a copy of the image
            annotated_image = black
            # Draw landmarks on face
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=self.results.multi_hand_landmarks[0],
                connections=self.mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(annotated_image, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

                    return cv2.flip(annotated_image, 1)

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()