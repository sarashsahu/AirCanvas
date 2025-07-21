import cv2
import numpy as np
import mediapipe as mp
import time
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QSlider, QHBoxLayout, QFrame,
                             QStatusBar, QFileDialog)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# Worker thread for computer vision processing
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.draw_color = (0, 0, 255)
        self.brush_thickness = 15
        self.eraser_thickness = 100
        self.current_tool = 'Draw'
        self.canvas = None
        self.last_frame = None

        self.xp, self.yp = 0, 0
        self.smooth_x, self.smooth_y = 0, 0
        self.alpha = 0.6

        self.ui_buttons = [
            {'name': 'Red', 'color': (0, 0, 255), 'pos': (20, 50)},
            {'name': 'Green', 'color': (0, 255, 0), 'pos': (20, 120)},
            {'name': 'Blue', 'color': (255, 0, 0), 'pos': (20, 190)},
            {'name': 'Eraser', 'color': (255, 255, 255), 'pos': (20, 260)}
        ]
        self.button_size = (100, 50)

    def draw_gesture_ui(self, image):
        for btn in self.ui_buttons:
            x, y = btn['pos']
            w, h = self.button_size
            cv2.rectangle(image, (x, y), (x + w, y + h), btn['color'], cv2.FILLED)
            
            is_selected = (self.current_tool == 'Draw' and self.draw_color == btn['color']) or \
                          (self.current_tool == 'Eraser' and btn['name'] == 'Eraser')
            if is_selected:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 4)

            text_color = (0, 0, 0)
            cv2.putText(image, btn['name'], (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)

            if self.canvas is None or self.canvas.shape[:2] != frame.shape[:2]:
                self.canvas = np.zeros_like(frame)

            self.last_frame = frame.copy()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            display_frame = frame.copy()

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                    index_tip = lm_list[8]
                    
                    self.smooth_x = int(self.alpha * index_tip[0] + (1 - self.alpha) * self.smooth_x)
                    self.smooth_y = int(self.alpha * index_tip[1] + (1 - self.alpha) * self.smooth_y)
                    smooth_tip = (self.smooth_x, self.smooth_y)

                    is_index_up = lm_list[8][1] < lm_list[6][1]
                    is_middle_up = lm_list[12][1] < lm_list[10][1]

                    if is_index_up and is_middle_up:
                        self.xp, self.yp = 0, 0
                        self.draw_gesture_ui(display_frame)
                        for btn in self.ui_buttons:
                            bx, by = btn['pos']
                            bw, bh = self.button_size
                            if bx < smooth_tip[0] < bx + bw and by < smooth_tip[1] < by + bh:
                                if btn['name'] == 'Eraser': self.set_tool_eraser()
                                else: self.set_tool_color(btn['color'])
                        cv2.circle(display_frame, smooth_tip, 20, (255, 255, 0), cv2.FILLED)

                    elif is_index_up:
                        cursor_color = (0,0,0) if self.current_tool == 'Eraser' else self.draw_color
                        cv2.circle(display_frame, smooth_tip, self.brush_thickness // 2, cursor_color, cv2.FILLED)

                        if self.xp == 0 and self.yp == 0: self.xp, self.yp = smooth_tip
                        
                        color = (0, 0, 0) if self.current_tool == 'Eraser' else self.draw_color
                        thickness = self.eraser_thickness if self.current_tool == 'Eraser' else self.brush_thickness
                        cv2.line(self.canvas, (self.xp, self.yp), smooth_tip, color, thickness)
                        self.xp, self.yp = smooth_tip
                    else:
                        self.xp, self.yp = 0, 0
            else:
                self.xp, self.yp = 0, 0

            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, inv_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
            inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

            display_frame = cv2.bitwise_and(display_frame, inv_mask)
            display_frame = cv2.bitwise_or(display_frame, self.canvas)

            self.change_pixmap_signal.emit(display_frame)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def save_drawing(self, with_background=False):
        if self.canvas is None: return None
        
        save_path, _ = QFileDialog.getSaveFileName(None, "Save Drawing", f"drawing_{time.strftime('%Y%m%d-%H%M%S')}.png", "PNG Images (*.png)")
        if not save_path: return None

        if with_background and self.last_frame is not None:
            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            background = cv2.bitwise_and(self.last_frame, self.last_frame, mask=cv2.bitwise_not(mask))
            final_image = cv2.add(background, self.canvas)
            cv2.imwrite(save_path, final_image)
        else:
            cv2.imwrite(save_path, self.canvas)
        return save_path

    def set_tool_color(self, color): self.current_tool = 'Draw'; self.draw_color = color
    def set_tool_eraser(self): self.current_tool = 'Eraser'
    def set_brush_size(self, value): self.brush_thickness = value
    def set_eraser_size(self, value): self.eraser_thickness = value
    def clear_canvas(self):
        if self.canvas is not None: self.canvas.fill(0)

class VirtualSlateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MayaSlate")
        self.setStyleSheet("""
            QMainWindow { background-color: #333; color: white; }
            QFrame { background-color: #444; border-radius: 5px; }
            QPushButton { background-color: #555; border: 1px solid #666; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
            QLabel { font-size: 14px; font-weight: bold; }
            QStatusBar { color: white; }
        """)
        self.resize(1280, 720) 

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setStatusBar(QStatusBar(self))

        controls_panel = QFrame()
        controls_layout = QVBoxLayout()
        controls_panel.setLayout(controls_layout)
        controls_panel.setFixedWidth(200)

        color_layout = QHBoxLayout()
        red_btn = QPushButton("Red"); red_btn.setStyleSheet("background-color: #c0392b;")
        green_btn = QPushButton("Green"); green_btn.setStyleSheet("background-color: #27ae60;")
        blue_btn = QPushButton("Blue"); blue_btn.setStyleSheet("background-color: #2980b9;")
        color_layout.addWidget(red_btn); color_layout.addWidget(green_btn); color_layout.addWidget(blue_btn)
        
        eraser_btn = QPushButton("Eraser")
        brush_slider = QSlider(Qt.Orientation.Horizontal); brush_slider.setRange(1, 50); brush_slider.setValue(15)
        eraser_slider = QSlider(Qt.Orientation.Horizontal); eraser_slider.setRange(20, 200); eraser_slider.setValue(100)
        
        clear_btn = QPushButton("Clear Canvas")
        save_with_bg_btn = QPushButton("Save with BG")
        save_without_bg_btn = QPushButton("Save without BG")
        quit_btn = QPushButton("Quit")

        controls_layout.addWidget(QLabel("Colors")); controls_layout.addLayout(color_layout)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Tools")); controls_layout.addWidget(eraser_btn)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Brush Size")); controls_layout.addWidget(brush_slider)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Eraser Size")); controls_layout.addWidget(eraser_slider)
        controls_layout.addSpacing(20)
        
        # NEW: Note for keyboard shortcuts
        note_label = QLabel(
            "<b>Note:</b><br>"
            "Press '<b>Q</b>' to Quit<br>"
            "Press '<b>E</b>' to Erase Canvas"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("font-size: 12px; font-weight: normal; color: #bbb;")
        controls_layout.addWidget(note_label)

        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(save_with_bg_btn)
        controls_layout.addWidget(save_without_bg_btn)
        controls_layout.addWidget(quit_btn)

        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(640, 360) 
        self.video_label.setStyleSheet("border: 2px solid #555; background-color: black;")
        main_layout.addWidget(controls_panel)
        main_layout.addWidget(self.video_label, 1)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        red_btn.clicked.connect(lambda: self.thread.set_tool_color((0, 0, 255)))
        green_btn.clicked.connect(lambda: self.thread.set_tool_color((0, 255, 0)))
        blue_btn.clicked.connect(lambda: self.thread.set_tool_color((255, 0, 0)))
        eraser_btn.clicked.connect(self.thread.set_tool_eraser)
        brush_slider.valueChanged.connect(self.thread.set_brush_size)
        eraser_slider.valueChanged.connect(self.thread.set_eraser_size)
        clear_btn.clicked.connect(self.thread.clear_canvas)
        save_with_bg_btn.clicked.connect(lambda: self.save_action(with_background=True))
        save_without_bg_btn.clicked.connect(lambda: self.save_action(with_background=False))
        quit_btn.clicked.connect(self.close)

    def save_action(self, with_background=False):
        filename = self.thread.save_drawing(with_background)
        if filename:
            self.statusBar().showMessage(f"Drawing saved to {filename}", 5000)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q: self.close()
        elif event.key() == Qt.Key.Key_E: self.thread.clear_canvas()
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VirtualSlateApp()
    window.show()
    sys.exit(app.exec())
