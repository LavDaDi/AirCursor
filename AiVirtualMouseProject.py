import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

wCam, hCam = 640, 480
frameR_left, frameR_right, frameR_top, frameR_bottom = 100, 100, 50, 150
smoothening = 4
pTime = 0
plocX, plocY, clocX, clocY = 0, 0, 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

# состояния кнопок
left_down = False
right_down = False

# состояния скролла
scroll_active = False
scroll_ref_y = None
dead_zone = 40   # мертвая зона (порог в пикселях)

# cooldown для навигации
last_nav_time = 0
nav_delay = 1.0  # секунда задержки

def is_scroll_gesture():
    """Жест-активатор: кулак + торчит мизинец (20) и большой палец (4)."""
    fingers = detector.fingersUp()
    return fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:  # рука в кадре
        # === Проверка скролл-жеста ===
        if is_scroll_gesture():
            if not scroll_active:
                scroll_active = True
                scroll_ref_y = lmList[0][2]  # y ладони (точка wrist)
                print("SCROLL MODE ACTIVATED")
            else:
                y_now = lmList[0][2]
                delta = y_now - scroll_ref_y

                if abs(delta) < dead_zone:
                    pass
                elif delta < -dead_zone:
                    pyautogui.scroll(30)
                elif delta > dead_zone:
                    pyautogui.scroll(-30)

            cv2.putText(img, "SCROLL MODE", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        else:
            if scroll_active:
                print("SCROLL MODE DEACTIVATED")
            scroll_active = False

            # === Курсор ===
            x1 = (lmList[4][1] + lmList[8][1]) // 2
            y1 = (lmList[4][2] + lmList[8][2]) // 2

            x3 = np.interp(x1, (frameR_left, wCam - frameR_right), (0, wScr))
            y3 = np.interp(y1, (frameR_top, hCam - frameR_bottom), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # === ЛЕВАЯ КНОПКА ===
            length_left, img, lineInfoL = detector.findDistance(4, 8, img)
            if length_left < 40 and not left_down and not right_down:
                pyautogui.mouseDown(button='left')
                left_down = True
            elif length_left >= 40 and left_down:
                pyautogui.mouseUp(button='left')
                left_down = False

            # === ПРАВАЯ КНОПКА ===
            length_right, img, lineInfoR = detector.findDistance(4, 12, img)
            if length_right < 20 and not right_down and not left_down:
                pyautogui.mouseDown(button='right')
                right_down = True
            elif length_right >= 20 and right_down:
                pyautogui.mouseUp(button='right')
                right_down = False

            # === НАВИГАЦИЯ (назад/вперед) ===
            current_time = time.time()

            # назад (большой + безымянный)
            length_back, img, _ = detector.findDistance(4, 16, img)
            if length_back < 30 and current_time - last_nav_time > nav_delay:
                pyautogui.hotkey("alt", "left")
                print("BACK")
                last_nav_time = current_time

            # вперед (большой + мизинец)
            length_forward, img, _ = detector.findDistance(4, 20, img)
            if length_forward < 30 and current_time - last_nav_time > nav_delay:
                pyautogui.hotkey("alt", "right")
                print("FORWARD")
                last_nav_time = current_time

            if left_down:
                cv2.circle(img, (lineInfoL[4], lineInfoL[5]), 15, (255, 0, 0), cv2.FILLED)
            if right_down:
                cv2.circle(img, (lineInfoR[4], lineInfoR[5]), 15, (0, 0, 255), cv2.FILLED)

    else:
        if scroll_active:
            print("SCROLL MODE DEACTIVATED")
        scroll_active = False

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
