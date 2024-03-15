""""
AI Virtual Mouse
Mon, 26/02/24 16:31H CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/
"""

import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


# Variables
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
threshold = 25
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Creating the detector object
detector = htm.handDetector(maxHands=1)

#Screen dimensions
wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

while True:

    # 1. Find the Hand Landmarks
    success, img = cap.read()
    # Find the hands
    img = detector.findHands(img)
    # Find the position of the hands, lmList give us the x and y position of the all landmarks
    lmList, bbox = detector.findPosition(img)

    # print(bbox)
    # 2. Tip of the index and middle fingers
    if len(lmList)!=0:
        # Points of the finger index
        x1, y1 = lmList[8][1:] # Point number 8, x and y position
        # Middel finger
        x2, y2 = lmList[4][1:] # Point number 12, x and y position
        # print(x1, y1, x2, y2)


        # 3. Check which fingers are up [1] if it is up, 0 if doesn't
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Only index finger: Moving Mode
        if fingers[1]==1 and fingers[2]==0:

            # 5. Convert Coordinates

            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # print(x1, x3)
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX)/ smoothening
            clocY = plocY + (y3 - plocY)/ smoothening
            # print(clocX)

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # Updating the values
            plocX, plocY = clocX, clocY

        # 8. Check if we are in clicking mode: Both index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[0] == 0:
            # 9. Find distance between fingers
            lenght, img, lineInfo = detector.findDistance(8, 4, img)
            print(lenght)
            # 10. Clock mouse if distance short
            if lenght < threshold:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)