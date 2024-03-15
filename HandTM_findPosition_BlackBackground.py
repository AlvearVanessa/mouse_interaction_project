"""
AI Virtual Mouse
Mon, 11/03/24 15:02H CET
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
mp_pose = mp.solutions.pose
mp_hand_mesh = mp.solutions.hands # .hands_connections


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


    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Create a face mesh object
        with self.mpHands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hand_mesh:
            black = np.zeros((350, 500, 3), dtype=np.uint8)
            results = hand_mesh.process(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))
        # Define boolean corresponding to whether or not a face was detected in the image
        hand_found = bool(results.multi_hand_landmarks)

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
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        lmList, bbox = detector.findPosition(img)
        #print(lmList)
        if len(lmList) != 0:
            print(lmList[4]) # Printing the x and y position of the landmaark 4

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()