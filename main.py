import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, PSM


VIDEO_SOURCE = './720p30fps.m4v'
VIDEO_SIZE = [1280, 720]
FRAMERATE = 30


def main():
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    frame_num = 0
    qte_keys_dict = dict()

    with PyTessBaseAPI(path='C:\\Program Files\\Tesseract-OCR\\tessdata', lang='eng') as tesseract_api:
        tesseract_api.SetVariable("tessedit_char_whitelist", "WwASsD")
        tesseract_api.SetPageSegMode(PSM.SINGLE_CHAR)

        while True:
            original_frame = video_capture.read()[1]
            grayscale_blurred_frame = cv2.medianBlur(cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY), 5)

            circles = get_hough_circles(grayscale_blurred_frame)

            qte_keys_dict = get_qte_keys_from_hough_circles(
                circles, original_frame, frame_num, qte_keys_dict, tesseract_api
            )
            qte_keys_list = create_qte_keys_list(qte_keys_dict, frame_num)

            place_qte_keys_text(original_frame, qte_keys_list)
            place_red_circles(circles, original_frame)

            cv2.imshow('Quick Time Event Detector', original_frame)
            if cv2.waitKey(round(1000 / FRAMERATE)) & 0xFF == ord('q'):
                break
            frame_num += 1

    close_window(video_capture)


def get_hough_circles(grayscale_blurred_frame):
    return cv2.HoughCircles(
        grayscale_blurred_frame,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=250,
        param2=55,
        minRadius=20,
        maxRadius=150,
    )


def get_qte_keys_from_hough_circles(circles, original_frame, frame_num, qte_keys_dict, tesseract_api):
    if circles is not None:
        for circle in circles[0]:
            qte_keys_dict = get_qte_keys_from_single_circle(
                circle, frame_num, original_frame, qte_keys_dict, tesseract_api
            )
    return qte_keys_dict


def get_qte_keys_from_single_circle(circle, frame_num, original_frame, qte_keys_dict, tesseract_api):
    x, y, radius = circle
    if 0 < x < VIDEO_SIZE[0] and 0 < y < VIDEO_SIZE[1]:
        cropped_frame = get_cropped_qte_frame(original_frame, radius, x, y, 0.65)
        processed_cropped_frame = get_processed_cropped_frame(cropped_frame)
        text = get_text_from_tesseract(processed_cropped_frame, tesseract_api)
        if text:
            qte_keys_dict[text.upper()] = frame_num
    return qte_keys_dict


def get_cropped_qte_frame(original_frame, radius, x, y, crop_percent):
    crop_radius = radius * crop_percent
    cropped_frame = original_frame[
        max(0, round(y - crop_radius)): min(VIDEO_SIZE[1], round(y + crop_radius)),
        max(0, round(x - crop_radius)): min(VIDEO_SIZE[0], round(x + crop_radius)),
    ]
    return cropped_frame


def get_processed_cropped_frame(cropped_frame):
    cropped_frame = cv2.resize(cropped_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    cropped_frame = cv2.GaussianBlur(cropped_frame, (9, 9), 0)
    cropped_frame = cv2.erode(cropped_frame, np.ones((2, 2), np.uint8), iterations=2)
    cropped_frame = cv2.threshold(cropped_frame, 127, 255, cv2.THRESH_BINARY)[1]
    return cropped_frame


def get_text_from_tesseract(processed_cropped_frame, tesseract_api):
    tesseract_api.SetImageBytes(processed_cropped_frame.tobytes(), *get_image_data(processed_cropped_frame))
    confidence_list = tesseract_api.AllWordConfidences()
    text = tesseract_api.GetUTF8Text()[:-1]
    if len(confidence_list) == 1 and confidence_list[0] >= .75 and len(text) == 1:
        return text


def get_image_data(processed_cropped_frame):
    height, width = processed_cropped_frame.shape[:2]
    bytes_per_pixel = processed_cropped_frame.shape[2] if len(processed_cropped_frame.shape) == 3 else 1
    bytes_per_line = bytes_per_pixel * width
    return width, height, bytes_per_pixel, bytes_per_line


def create_qte_keys_list(qte_keys_dict, frame_num):
    qte_keys_list = []
    for key, frame in qte_keys_dict.items():
        if frame_num - 10 <= frame <= frame_num:
            qte_keys_list.append(key)
    return qte_keys_list


def place_qte_keys_text(original_frame, qte_keys_list):
    cv2.rectangle(original_frame, (0, 0), (300, 50), (0, 0, 0), -1)
    cv2.putText(
        img=original_frame,
        text='Detected QTE Keys: ' + ', '.join(qte_keys_list),
        org=(30, 30),
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


main()
