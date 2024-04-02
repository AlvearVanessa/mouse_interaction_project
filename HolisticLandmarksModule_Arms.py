"""
Thur, 02/04/24 11:54H CET
Holistic Module
Website: https://www.computervision.zone/
https://omes-va.com/mediapipe-holistic-mediapipe-python/
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np


mp_pose_mesh = mp.solutions.pose

class PoseDetector():
    def __init__(self, mode=False, modelC=1, smoothLmks=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelC = modelC
        self.smoothLmks = smoothLmks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.Poses = self.mpPose.Pose(self.mode, self.modelC, self.smoothLmks,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose_landmarks.process(imgRGB)
        # print(results.multi_hand_landmarks)


        #results1 = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pose_found = bool(self.results.pose_landmarks)
        print(pose_found)
        if pose_found==True:

            for poseLms in self.results.pose_landmarks:
                if draw:
                    black = np.zeros((350, 500, 3), dtype=np.uint8)
                    annotated_image = black
                    self.mpDraw.draw_landmarks(annotated_image, poseLms,
                                               self.mpPose.POSE_CONNECTIONS)

                    return annotated_image


    def findHands2(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose_landmarks.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.pose_landmarks:
            for handType, poseLms in zip(self.results.multi_handedness, self.results.pose_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(poseLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, poseLms,
                                               self.mpPose.POSE_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return allHands, img

    def findPoseLandmarks(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose_landmarks.process(imgRGB)
        # print(results.multi_hand_landmarks)
        poses = []
        h, w, c = img.shape
        # results1 = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.results.pose_landmarks:
            for poseLms in self.results.self.results.pose_landmarks:
                if draw:
                    # black = np.zeros((350, 500, 3), dtype=np.uint8)
                    # annotated_image = black
                    self.mpDraw.draw_landmarks(img, poseLms,
                                               self.mpPose.POSE_CONNECTIONS)
                pose = []
                for id, lm in enumerate(poseLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    pose.append([x, y])
                poses.append(pose)
        return img, poses

    def findPosePosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.pose_landmarks:
            myPose= self.results.pose_landmarks[handNo]
            for id, lm in enumerate(myPose.landmark):
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
            # bbox = xmin, ymin, xmax, ymax
            bbox.append(xmin)
            bbox.append(ymin)
            bbox.append(xmax)
            bbox.append(ymax)

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):

        #results1 = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pose_found = bool(self.results.pose_landmarks)
        if pose_found==True:
            fingers = []
            # Thumb
            if self.lmList[22][1] > self.lmList[22 - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if self.lmList[22][2] < self.lmList[22 - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # totalFingers = fingers.count(1)

            return fingers# , self.lmList[self.tipIds[0]])


    def findPoseDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def wrist(self):
        self.lmList[16] == 0
        return print(self.lmList[0][1:])






def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        if success==True:
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)
            print(bbox)
            if len(lmList) != 0:
                print(lmList[4])

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()