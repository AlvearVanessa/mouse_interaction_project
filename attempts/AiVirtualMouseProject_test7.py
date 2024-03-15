""""
AI Virtual Mouse
Mon, 26/020/24 16:31H CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/
"""

import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import mouse


# Variables
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
threshold_click = 70
threshold_scroll_up = 35
threshold_scroll_down= 55
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
dimy = 400
LRangeBlk = np.array([0,0,0])
URangeBlk = np.array([2,2,2])

#print(wScr, hScr)

## Other params
dimx = 750
# BGR
RED = (0, 0, 255)

def showInMovedWindow(winname, imge, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,imge)
    cv2.waitKey(1)


# Def a functin to create a mask for certain color range
#def gatherPoints(frame, lowerRange, upperRange):
#    mask =cv2.inRange(frame, lowerRange, upperRange)




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
        x1, y1 = lmList[5][1:] # Point number 5, x and y position
        # Middel finger
        x2, y2 = lmList[9][1:] # Point number 9, x and y position
        # print(x1, y1, x2, y2)


        # 3. Check which fingers are up [1] if it is up, 0 if doesn't
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Only index finger: Moving Mode
        if fingers[1]==1:

            # 5. Convert Coordinates

            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # lenght0, img, _ = detector.findDistance(5, 9, img)
            # print("Moving Mode distance lmark 5 and 9 = ", lenght0)
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
        if fingers[2]== 0 and fingers[3]== 0 and fingers[4] == 0:
            # 9. Find distance between fingers
            lenght, img, lineInfo = detector.findDistance(7, 5, img)
            print("Clicking mode len lmark 7 and 5 = ", lenght)
            # 10. Click mouse if distance short
            if lenght < threshold_click:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 12. Mouse scrolling bottom up
        if fingers[1] == 1 and fingers[2] == 1:
            # 13. Find distance between fingers
            lenght2, img, lineInfo2 = detector.findDistance(8, 12, img)
            lenght4, img, lineInfo4 = detector.findDistance(12, 9, img)
            # print("Mouse scrolling UP dist 8 and 12= ", lenght2)
            print("Mouse scrolling UP dist 12 and 9= ", lenght4)
            # 14. Clock mouse if distance short
            if lenght2 < threshold_scroll_up :
                cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                           15, (255, 0, 0), cv2.FILLED)
                mouse.wheel(delta=1)


        # 15. Mouse scrolling bottom down
        #if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # 16. Find distance between fingers
        #lenght3, img, lineInfo3 = detector.findDistance(12, 0, img)
        #print("Mouse scrolling DOWN 12 and 0 = ",lenght3)
        # 17. Click mouse if distance short
        #if lenght3 < threshold_scroll_down:
                #    cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
            #               15, (255, 189, 25), cv2.FILLED)
        #    mouse.wheel(delta=-1)



    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    # cv2.imshow("Image", img, extent=[100, 100, 100, 100])
    #cv2.waitKey(1)

    _, mask = cv2.threshold(img, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    im_thresh_gray = cv2.bitwise_and(img, mask)
    # mask2 = np.zeros(img.shape[:2], dtype="uint8")
    showInMovedWindow('Image example', im_thresh_gray, 0, 500)
    #mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channel mask
    #showInMovedWindow('Image example', mask3, 0, 500)

# Source https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image

