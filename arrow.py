import itertools
from operator import itemgetter
import cv2
import numpy as np


def get_arrow(cropped_frame):
    corner_coords_list = get_corner_coords(cropped_frame)

    for coords in itertools.product(corner_coords_list, repeat=3):
        coords = sorted(coords, key=itemgetter(0))
        if is_arrow__centre_to_right_and_clockwise_to_left(coords, cropped_frame):
            return "Centre to right and clockwise to left"
        elif is_arrow__shift_key(coords, cropped_frame):
            return "Shift"


def get_corner_coords(cropped_frame):
    grayscale_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.dilate(cv2.cornerHarris(grayscale_frame, 2, 3, 0.04), None)
    threshold = np.uint8(cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1])
    centroids = cv2.connectedComponentsWithStats(threshold)[3]
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001
    corner_coords = cv2.cornerSubPix(grayscale_frame, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return corner_coords


def is_arrow__centre_to_right_and_clockwise_to_left(coord_set, cropped_frame):
    height, width = cropped_frame.shape[:2]
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        (x1, y1) != (x2, y2) and (x2, y2) != (x3, y3) and (x1, y1) != (x3, y3) and  # Coords not equal
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.15 <= x3 - x1 <= height * 0.25 and  # Width
        height * 0.075 <= y1 - y2 <= height * 0.12 and  # Height
        x3 < width / 2.5 and  # Left side
        y2 < height / 2  # Top side
    )


def is_arrow__shift_key(coord_set, cropped_frame):
    height, width = cropped_frame.shape[:2]
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        (x1, y1) != (x2, y2) and (x2, y2) != (x3, y3) and (x1, y1) != (x3, y3) and  # Coords not equal
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.4 <= x3 - x1 <= height * 0.48 and  # Width
        height * 0.2 <= y1 - y2 <= height * 0.24 and  # Height
        width / 4 < x3 < 3 * width / 4 and  # Middle x-axis
        y2 < 3 * height / 5  # Top side
    )
