import tkinter as tk
from tkinter import filedialog
import cv2
import time
import numpy as np
from tesserocr import PyTessBaseAPI, PSM
from symbol import get_symbol
from text import get_text_from_tesseract


def main():
    root = tk.Tk()
    root.withdraw()
    video_source = filedialog.askopenfilename(
        title='Open video recording from "Detroit: Become Human"',
        filetypes=[('Video file', '.mp4 .m4v .mkv')],
    )

    if video_source:
        handle_video_source(video_source)


def handle_video_source(video_source):
    video_capture = cv2.VideoCapture(video_source)
    framerate = video_capture.get(cv2.CAP_PROP_FPS)
    qte_dict = dict()

    with PyTessBaseAPI(path='C:\\Program Files\\Tesseract-OCR\\tessdata', lang='eng') as tesseract_api:
        tesseract_api.SetVariable('tessedit_char_whitelist', 'WwASsDE')
        tesseract_api.SetPageSegMode(PSM.SINGLE_CHAR)
        loop_each_frame(qte_dict, tesseract_api, video_capture, framerate)

    close_window(video_capture)


def loop_each_frame(qte_dict, tesseract_api, video_capture, framerate):
    while True:
        start_time = time.time()

        ret, original_frame = video_capture.read()
        if not ret:
            break
        grayscale_blurred_frame = cv2.medianBlur(cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY), 5)

        circles = get_hough_circles(grayscale_blurred_frame)

        frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        qte_dict = get_qte_dict_from_hough_circles(circles, original_frame, frame_num, qte_dict, tesseract_api)
        qte_list = create_qte_list(qte_dict, frame_num)

        place_qte_text(original_frame, qte_list)
        place_red_circles(circles, original_frame)

        cv2.imshow('Quick Time Event Detector', original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        while end_time - start_time < 1 / framerate:
            end_time = time.time()


def get_hough_circles(grayscale_blurred_frame):
    height = grayscale_blurred_frame.shape[0]
    return cv2.HoughCircles(
        grayscale_blurred_frame,
        cv2.HOUGH_GRADIENT,
        1,
        round(height * 0.075),
        param1=250,
        param2=55,
        minRadius=round(height * 0.025),
        maxRadius=round(height * 0.125),
    )


def get_qte_dict_from_hough_circles(circles, original_frame, frame_num, qte_dict, tesseract_api):
    if circles is not None:
        for circle in circles[0]:
            qte_dict = get_qte_dict_from_single_circle(circle, frame_num, original_frame, qte_dict, tesseract_api)
    return qte_dict


def get_qte_dict_from_single_circle(circle, frame_num, original_frame, qte_dict, tesseract_api):
    x, y, radius = circle
    height, width = original_frame.shape[:2]
    if 0 < x < width and 0 < y < height:
        cropped_frame = get_cropped_qte_frame(original_frame, height, width, radius, x, y, 0.65)
        text = get_text_from_tesseract(cropped_frame, tesseract_api)
        if text:
            qte_dict[text.upper()] = frame_num
        else:
            cropped_frame_small = get_cropped_qte_frame(original_frame, height, width, radius, x, y, 1.25)
            cropped_frame_large = get_cropped_qte_frame(original_frame, height, width, radius, x, y, 1.75)
            symbol = get_symbol(cropped_frame_small, cropped_frame_large)
            if symbol:
                qte_dict[symbol] = frame_num
    return qte_dict


def get_cropped_qte_frame(original_frame, height, width, radius, x, y, crop_percent):
    crop_radius = radius * crop_percent
    cropped_frame = original_frame[
        max(0, round(y - crop_radius)): min(height, round(y + crop_radius)),
        max(0, round(x - crop_radius)): min(width, round(x + crop_radius)),
    ]
    return cropped_frame


def create_qte_list(qte_dict, frame_num):
    qte_list = []
    for key, frame in qte_dict.items():
        if frame_num - 10 <= frame <= frame_num:
            qte_list.append(key)
    return qte_list


def place_qte_text(original_frame, qte_list):
    cv2.rectangle(original_frame, (0, 0), (465, 80), (0, 0, 0), -1)
    keys = ', '.join(filter(lambda x: len(x) == 1 or x == 'Shift' or x == 'Space', qte_list))
    gestures = ', '.join(filter(lambda x: len(x) != 1 and x != 'Shift' and x != 'Space', qte_list))
    place_text(original_frame, 'Key(s): ' + (keys if len(keys) > 0 else 'None'), 30)
    place_text(original_frame, 'Gesture(s): ' + (gestures if len(gestures) > 0 else 'None'), 60)


def place_text(original_frame, qte_text, y_pos):
    cv2.putText(
        img=original_frame,
        text=qte_text,
        org=(20, y_pos),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
    )


def place_red_circles(circles, original_frame):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, radius in circles[0]:
            cv2.circle(original_frame, (x, y), radius, (0, 0, 255), 2)


def close_window(video_capture):
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
