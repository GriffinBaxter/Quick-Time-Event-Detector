import itertools
from operator import itemgetter
import cv2
import numpy as np


def get_symbol(cropped_frame_small, cropped_frame_large):
    corner_coords_small = get_corner_coords(cropped_frame_small)
    height, width = cropped_frame_small.shape[:2]

    for coords in itertools.combinations(corner_coords_small, r=3):
        coords = sorted(coords, key=itemgetter(0))
        if is_centre_to_right_and_clockwise_to_left(coords, height, width):
            return "Centre to right and clockwise to left"
        elif is_shift_key(coords, height, width):
            return "Shift"

    for coords in itertools.combinations(corner_coords_small, r=2):
        coords = sorted(coords, key=itemgetter(0))
        if is_space_key(coords, height, width):
            return "Space"

    corner_coords_large = get_corner_coords(cropped_frame_large)
    height, width = cropped_frame_large.shape[:2]

    for coords in itertools.combinations(corner_coords_large, r=3):
        coords = sorted(coords, key=itemgetter(1))
        if is_left(coords, height, width):
            return "Left"
        elif is_right(coords, height, width):
            return "Right"
        coords = sorted(coords, key=itemgetter(0))
        if is_up(coords, height, width):
            return "Up"
        elif is_down(coords, height, width):
            return "Down"


def get_corner_coords(cropped_frame):
    grayscale_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.dilate(cv2.cornerHarris(grayscale_frame, 2, 3, 0.04), None)
    threshold = np.uint8(cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1])
    centroids = cv2.connectedComponentsWithStats(threshold)[3]
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001
    corner_coords = cv2.cornerSubPix(grayscale_frame, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return corner_coords


def is_centre_to_right_and_clockwise_to_left(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.15 <= x3 - x1 <= height * 0.25 and  # Width
        height * 0.075 <= y1 - y2 <= height * 0.12 and  # Height
        x3 < width / 2.5 and  # Left side
        y2 < height / 2  # Top side
    )


def is_shift_key(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.4 <= x3 - x1 <= height * 0.48 and  # Width
        height * 0.2 <= y1 - y2 <= height * 0.24 and  # Height
        width / 4 < x3 < 3 * width / 4 and  # Middle x-axis
        y2 < 3 * height / 5  # Top side
    )


def is_space_key(coord_set, height, width):
    (x1, y1), (x2, y2) = coord_set
    return (
        0 <= abs(y1 - y2) <= height * 0.04 and  # Horizontal bottom
        height * 0.25 <= x2 - x1 <= height * 0.3 and  # Width
        width * 0.48 < (x1 + x2) / 2 < width * 0.52 and  # Middle x-axis
        height * 0.52 < (y1 + y2) / 2 < height * 0.54  # Just below y-axis
    )


def is_left(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(x1 - x3) <= height * 0.04 and  # Vertical bottom
        0 <= abs((y2 - y1) - (y3 - y2)) <= height * 0.04 and  # Top in centre
        height * 0.05 <= x1 - x2 <= height * 0.1 and  # Width
        height * 0.175 <= y3 - y1 <= height * 0.225 and  # Height
        x1 < 0.175 * width and  # Left side
        height * 0.45 < y2 < height * 0.55  # Middle y-axis
    )


def is_right(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(x1 - x3) <= height * 0.04 and  # Vertical bottom
        0 <= abs((y2 - y1) - (y3 - y2)) <= height * 0.04 and  # Top in centre
        height * 0.05 <= x2 - x1 <= height * 0.1 and  # Width
        height * 0.175 <= y3 - y1 <= height * 0.225 and  # Height
        0.825 * width < x1 and  # Right side
        height * 0.45 < y2 < height * 0.55  # Middle y-axis
    )


def is_up(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.175 <= x3 - x1 <= height * 0.225 and  # Width
        height * 0.05 <= y1 - y2 <= height * 0.1 and  # Height
        y1 < 0.175 * width and  # Top side
        height * 0.45 < x2 < height * 0.55  # Middle x-axis
    )


def is_down(coord_set, height, width):
    (x1, y1), (x2, y2), (x3, y3) = coord_set
    return (
        0 <= abs(y1 - y3) <= height * 0.04 and  # Horizontal bottom
        0 <= abs((x2 - x1) - (x3 - x2)) <= height * 0.04 and  # Top in centre
        height * 0.175 <= x3 - x1 <= height * 0.225 and  # Width
        height * 0.05 <= y2 - y1 <= height * 0.1 and  # Height
        0.825 * width < y1 and  # Bottom side
        height * 0.45 < x2 < height * 0.55  # Middle x-axis
    )
