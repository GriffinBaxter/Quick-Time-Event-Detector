import cv2
import numpy as np


def get_text_from_tesseract(cropped_frame, tesseract_api):
    processed_cropped_frame = get_processed_cropped_frame(cropped_frame)
    tesseract_api.SetImageBytes(processed_cropped_frame.tobytes(), *get_image_data(processed_cropped_frame))
    confidence_list = tesseract_api.AllWordConfidences()
    text = tesseract_api.GetUTF8Text()[:-1]
    if len(confidence_list) == 1 and confidence_list[0] >= .75 and len(text) == 1:
        return text


def get_processed_cropped_frame(cropped_frame):
    cropped_frame = cv2.resize(cropped_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    cropped_frame = cv2.GaussianBlur(cropped_frame, (9, 9), 0)
    cropped_frame = cv2.erode(cropped_frame, np.ones((2, 2), np.uint8), iterations=2)
    cropped_frame = cv2.threshold(cropped_frame, 127, 255, cv2.THRESH_BINARY)[1]
    return cropped_frame


def get_image_data(processed_cropped_frame):
    height, width = processed_cropped_frame.shape[:2]
    bytes_per_pixel = processed_cropped_frame.shape[2] if len(processed_cropped_frame.shape) == 3 else 1
    bytes_per_line = bytes_per_pixel * width
    return width, height, bytes_per_pixel, bytes_per_line
