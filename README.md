# Quick-Time Event Detector

The detector is able to find quick-time events and display the action that they represent, using footage from the PC
version of the game 'Detroit: Become Human' by Quantic Dream.

## Code Structure Overview

The code is structured into `main`, `text`, and `symbol` packages.

### main

This is the main package of the codebase. This includes the pop-up for selecting a video file, the iteration through
each video frame, finding Hough circles, and the placing of text and circles on the frame.

### text

This package contains the Tesseract OCR text detection. This includes the frame pre-processing, and execution of the
Tesseract API for each frame.

### symbol

This package contains the symbol detection. This includes finding Harris corners, and using combinations of their
coordinates to detect symbols.

## Running the Detector

### Package Requirements

The following Python packages are required to run the Quick-Time Event Detector:
* tkinter
* cv2 (OpenCV)
* numpy
* tesserocr (Tesseract OCR)

### How To Run

1. Update the `TESSDATA_PATH` variable if required (should be set to the path of the `tessdata` folder for Tesseract
OCR).
2. Run `main.py`.
3. Select a video file containing footage from the PC version of 'Detroit: Become Human'.
