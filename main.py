import tkinter as tk
from tkinter import filedialog
import cv2
import time
import numpy as np
from tesserocr import PyTessBaseAPI, PSM
from symbol import get_symbol
from text import get_text_from_tesseract


# Path of the tessdata folder for Tesseract OCR
TESSDATA_PATH = 'C:\\Program Files\\Tesseract-OCR\\tessdata'


def main():
    """
    Main function to start running the Quick Time Event Detector. Opens a window to select a video file, and starts the
    process if a video source is found.
    """
    root = tk.Tk()
    root.withdraw()
    video_source = filedialog.askopenfilename(
        title='Open video recording from "Detroit: Become Human"',
        filetypes=[('Video file', '.mp4 .m4v .mkv')],
    )

    if video_source:
        handle_video_source(video_source)


def handle_video_source(video_source):
    """
    Takes the video source and its framerate, sets up Tesseract OCR, and calls the function to loop each video frame.
    :param video_source: The selected video source.
    """
    video_capture = cv2.VideoCapture(video_source)
    framerate = video_capture.get(cv2.CAP_PROP_FPS)
    qte_dict = dict()

    with PyTessBaseAPI(path=TESSDATA_PATH, lang='eng') as tesseract_api:
        tesseract_api.SetVariable('tessedit_char_whitelist', 'WwASsDE')
        tesseract_api.SetPageSegMode(PSM.SINGLE_CHAR)
        loop_each_frame(qte_dict, tesseract_api, video_capture, framerate)

    close_window(video_capture)


def loop_each_frame(qte_dict, tesseract_api, video_capture, framerate):
    """
    Conducts the QTE detection for each frame. This includes reading the frame, grayscaling and blurring for finding
    Hough circles, retrieving the QTE dictionary and list, placing text and circles on the frame, and changing the frame
    if enough time has already passed (to stabilise the frametimes).
    :param qte_dict: The QTE dictionary.
    :param tesseract_api: Tesseract OCR API.
    :param video_capture: The video capture.
    :param framerate: Framerate.
    :return:
    """
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
    """
    Finds Hough circles within the given grayscaled and blurred frame.
    :param grayscale_blurred_frame: A grayscaled and blurred frame.
    :return: The Hough circles found (if any).
    """
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
    """
    Updates the QTE dictionary for each Hough circle.
    :param circles: Detected Hough circles.
    :param original_frame: The original (unmodified) frame.
    :param frame_num: Frame number.
    :param qte_dict: The QTE dictionary.
    :param tesseract_api: Tesseract OCR API.
    :return: The QTE dictionary (may or may not be updated).
    """
    if circles is not None:
        for circle in circles[0]:
            qte_dict = get_qte_dict_from_single_circle(circle, frame_num, original_frame, qte_dict, tesseract_api)
    return qte_dict


def get_qte_dict_from_single_circle(circle, frame_num, original_frame, qte_dict, tesseract_api):
    """
    Updates the QTE dictionary if a QTE is found within the given frame's circle, by attempting to detect text or a
    symbol.
    :param circle: Detected Hough circle.
    :param frame_num: Frame number.
    :param original_frame: The original (unmodified) frame.
    :param qte_dict: The QTE dictionary.
    :param tesseract_api: Tesseract OCR API.
    :return: The QTE dictionary (may or may not be updated).
    """
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
    """
    Crops a frame using the given coordinates and dimensions.
    :param original_frame: The original (unmodified) frame.
    :param height: The frame's height.
    :param width: The frame's width.
    :param radius: The crop radius.
    :param x: x-axis crop centre position.
    :param y: y-axis crop centre position.
    :param crop_percent: The crop percent (applied to the radius).
    :return: The cropped frame.
    """
    crop_radius = radius * crop_percent
    cropped_frame = original_frame[
        max(0, int(round(y - crop_radius))): min(height, int(round(y + crop_radius))),
        max(0, int(round(x - crop_radius))): min(width, int(round(x + crop_radius))),
    ]
    return cropped_frame


def create_qte_list(qte_dict, frame_num):
    """
    Creates a list of the currently detected QTE keys based on the last 10 frames.
    :param qte_dict: The QTE dictionary.
    :param frame_num: Frame number.
    :return: The QTE List.
    """
    qte_list = []
    for key, frame in qte_dict.items():
        if frame_num - 10 <= frame <= frame_num:
            qte_list.append(key)
    return qte_list


def place_qte_text(original_frame, qte_list):
    """
    Places text of the detected keys on top of a black rectangle.
    :param original_frame: The original (unmodified) frame.
    :param qte_list: The QTE List.
    """
    cv2.rectangle(original_frame, (0, 0), (465, 80), (0, 0, 0), -1)
    keys = ', '.join(filter(lambda x: len(x) == 1 or x == 'Shift' or x == 'Space', qte_list))
    gestures = ', '.join(filter(lambda x: len(x) != 1 and x != 'Shift' and x != 'Space', qte_list))
    place_text(original_frame, 'Key(s): ' + (keys if len(keys) > 0 else 'None'), 30)
    place_text(original_frame, 'Gesture(s): ' + (gestures if len(gestures) > 0 else 'None'), 60)


def place_text(original_frame, qte_text, y_pos):
    """
    Places text on a frame for a given y-axis position.
    :param original_frame: The original (unmodified) frame.
    :param qte_text: The QTE text.
    :param y_pos: The y-axis position.
    """
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
    """
    Places red circles on each of the detected Hough circles (if any).
    :param circles: Detected Hough circles.
    :param original_frame: The original (unmodified) frame.
    """
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, radius in circles[0]:
            cv2.circle(original_frame, (x, y), radius, (0, 0, 255), 2)


def close_window(video_capture):
    """
    Closes the QTE detector window.
    :param video_capture: The video capture.
    """
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
