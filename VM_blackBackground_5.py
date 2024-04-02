""""
AI Virtual Mouse
Mon, 18/03/24 16:37H CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/
"""

import cv2
import numpy as np
#import HandTrackingModule_black3 as htm
import HandTrackingModule_black_camera_hand_distance as htm
import time
import math
import autopy
import mouse


##########################
wCam, hCam = 1280, 720
frameR = 80  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
# time.sleep(2)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

threshold_click = 40
threshold_scroll_up = 22
threshold_scroll_down= 50
smoothening = 7

# Find Function
# x is the raw distance y is the value in cm
x_ = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y_ = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x_, y_, 2)  # y = Ax^2 + Bx + C




def showInMovedWindow(winname, imge, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,imge)



while True:
    # 1. Find hand Landmarks
    success, img = cap.read()

    if success==True:

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # print rectangle to resize screen
        cv2.rectangle(img, (frameR, frameR), (wCam - 11*frameR, hCam - (5*frameR+50)),
                      (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - 11*frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - 5*frameR), (0, hScr+200))
            # 6. Smoothen Values (to avoid the mouse checking)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # print("Coordinates ", clocX, clocY)

            # 7. Move Mouse
            #length0, img, lineInfo = detector.findDistance(8, 12, img)
            #print("Mouse Moving Mode dist 8 and 12 = ", length0)
            #if length0 > threshold_click:
            autopy.mouse.move(wScr - clocX, clocY)
            # print("Screen dimensions", wScr - clocX, clocY)
            #scale = autopy.screen.scale()
            # autopy.mouse.smooth_move(1700 / scale, 0 / scale)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Check if we are in clicking mode: Both index and middle fingers are up: Clicking Mode
        lenght, img, lineInfo = detector.findDistance(12, 0, img)
        #print("Clicking mode len 12 and 0 = ", lenght)
        # 10. Click mouse if distance short
        if fingers[1] == 0 and fingers[0] == 1:
            if 20< lenght < 47:
            # 9. Find distance between fingers
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                time.sleep(0.75) # 0.75

        # 12. Mouse scrolling bottom up
        if fingers[1] == 1 and fingers[2]==1 :
            # 13. Find distance between fingers
            lenght2, img, lineInfo2 = detector.findDistance(8, 12, img)
            #print("Mouse scrolling UP len 8 and 12 = ", lenght2)
            # 14. Clock mouse if distance short
            if lenght2 < threshold_scroll_up:
                cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                           15, (255, 0, 0), cv2.FILLED)
                mouse.wheel(delta=1)


        # 12. Mouse scrolling bottom DOWN
        if fingers[1] == 0 and fingers[2]==0 :
            # 13. Find distance between fingers
            lenght3, img, lineInfo3 = detector.findDistance(8, 12, img)
            #print("Mouse scrolling UP len 8 and 12 = ", lenght2)
            # 14. Clock mouse if distance short
            if lenght3 < threshold_scroll_down:
                cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
                           15, (255, 0, 0), cv2.FILLED)
                mouse.wheel(delta=-1)

        lm_list, bbox = detector.findPosition(img)
        x, y, w, h = bbox
        # print(lm_list[5][1:])
        # x and y position for landmk 5 and 17
        x1, y1 = lm_list[5][1:]
        x2, y2 = lm_list[17][1:]
        print(x1, x2, y1, y2)
        distance = int(math.sqrt((abs(x2 - x1)) ** 2 + (abs(y2 - y1)) ** 2))
        print(distance)


        A, B, C = coff
        distanceCM1 = A * distance ** 2 + B * distance + C
        print(distanceCM1, distance)


            # cv2.rectangle(img, (x, y), (w + 10, h + 10), (255, 0, 255), 3)
            # putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))

        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.putText(img, f'{int(distanceCM1)} cm', (x + 5, y - 10), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # scale = autopy.screen.scale()
        # autopy.mouse.smooth_move(1700 / scale, 0 / scale)





        # 12. Display
        # 610 is for put the imshow in the inferior left corner
        showInMovedWindow('Image example', img, 0, 610)
        # To put the imshow image in front of each page - anytime
        cv2.setWindowProperty('Image example', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1) & 0xFF == ord('q')

    else:
        break

cap.release()

