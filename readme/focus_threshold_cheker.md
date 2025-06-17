# About the focus threshold
This script is designed to assist in determining a threshold for focus-based filtering in side-profile detection tasks. By analyzing a video file, it detects head regions using a YOLO model, classifies side-profile images using a CNN, evaluates image sharpness using variance-based focus scores, and visually presents images across different focus levels. The user can then visually inspect these examples to select an appropriate threshold to separate blurry images from sharp ones.

Use the displayed images and their associated focus scores as reference samples to help determine a suitable blur threshold. Choose a value above the scores of noticeably blurry images.

```
python focus_threshold_checker.py -f $movie -n 15
```
-f or --movie_path : Path to the video file (※ webcam input is not supported)

-n or --number : Number of focus levels (images) to extract for threshold reference

Example:
python focus_threshold_checker.py -f ./video/sample.mp4 -n 10

This script analyzes side-profile images extracted from a video and displays them across multiple levels of focus variance to help determine a sharpness threshold.
Note: Webcam input is not supported.