import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

wCam, hCam = 640, 480  # Разрешение камеры
frameR_left = 100  # Отступ слева
frameR_right = 100  # Отступ справа
frameR_top = 50  # Отступ сверху
frameR_bottom = 150  # Отступ снизу
smoothening = 4  # Сглаживание движения
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    if len(lmList) != 0:
        # Берём середину между большим (4) и указательным (8)
        x1 = (lmList[4][1] + lmList[8][1]) // 2
        y1 = (lmList[4][2] + lmList[8][2]) // 2

        x2, y2 = lmList[12][1:]
        
        fingers = detector.fingersUp()
        # Рисуем прямоугольник с индивидуальными отступами
        cv2.rectangle(img, (frameR_left, frameR_top), (wCam - frameR_right, hCam - frameR_bottom),
                      (255, 0, 255), 2)

        # Движение мышки
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR_left, wCam - frameR_right), (0, wScr))
            y3 = np.interp(y1, (frameR_top, hCam - frameR_bottom), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Клик мышки
        if fingers[1] == 1 and fingers[0] == 1:  # указательный и большой подняты
            length, img, lineInfo = detector.findDistance(4, 8, img)

            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # Отображение FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()