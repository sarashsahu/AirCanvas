import cv2
import numpy as np
import mediapipe as mp
import time

# Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Drawing variables
draw_color = (0, 0, 255)  # Default: red
brush_thickness = 10
eraser_thickness = 50
xp, yp = 0, 0
canvas = None
current_tool = 'Draw'

# Color buttons layout
color_buttons = [
    {'color': (0, 0, 255), 'name': 'Red'},
    {'color': (0, 255, 0), 'name': 'Green'},
    {'color': (255, 0, 0), 'name': 'Blue'},
    {'color': (0, 0, 0), 'name': 'Eraser'}
]

def draw_ui_panel(img, current_tool):
    panel_width = 150
    cv2.rectangle(img, (img.shape[1] - panel_width, 0), (img.shape[1], img.shape[0]), (30, 30, 30), -1)

    y = 20
    for button in color_buttons:
        color = button['color']
        name = button['name']
        is_selected = (draw_color == color) if name != 'Eraser' else (current_tool == 'Eraser')
        thickness = 3 if is_selected else 1
        cv2.rectangle(img, (img.shape[1] - panel_width + 10, y), (img.shape[1] - 10, y + 50), color, -1)
        cv2.rectangle(img, (img.shape[1] - panel_width + 10, y), (img.shape[1] - 10, y + 50), (255, 255, 255), thickness)
        cv2.putText(img, name, (img.shape[1] - panel_width + 15, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 60

    # Show instructions
    cv2.putText(img, "S: Save", (img.shape[1] - panel_width + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    cv2.putText(img, "C: Clear", (img.shape[1] - panel_width + 10, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    cv2.putText(img, "Q: Quit", (img.shape[1] - panel_width + 10, y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

def check_button_selection(x, y, width, height):
    panel_x = width - 150
    btn_y = 20
    for button in color_buttons:
        if panel_x + 10 < x < width - 10 and btn_y < y < btn_y + 50:
            return button['name']
        btn_y += 60
    return None

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            index_tip = lm_list[8]
            middle_tip = lm_list[12]

            x1, y1 = index_tip
            x2, y2 = middle_tip

            fingers_up = y1 < lm_list[6][1] and y2 < lm_list[10][1]  # Both fingers up

            if fingers_up:
                selected = check_button_selection(x1, y1, frame.shape[1], frame.shape[0])
                if selected:
                    if selected == 'Eraser':
                        current_tool = 'Eraser'
                    else:
                        draw_color = [c for c in color_buttons if c['name'] == selected][0]['color']
                        current_tool = 'Draw'
                xp, yp = 0, 0
            else:
                # Drawing mode
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if current_tool == 'Eraser':
                    cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    draw_ui_panel(frame, current_tool)

    cv2.imshow("Virtual Slate", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"drawing_{timestamp}.png", canvas)
        print(f"[INFO] Drawing saved as drawing_{timestamp}.png")

cap.release()
cv2.destroyAllWindows()
