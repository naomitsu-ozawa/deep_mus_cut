# %%
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ultralytics import YOLO

from moucut_tools import kmeans

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# %%
def moucut(movie_path, decive_flag, image_flag, show_flag):
    if movie_path == "webcam":
        movie_path = 0

    device_name = decive_flag
    print("runing_TF_version")
    print(f"物体検出に{device_name}を利用します。")
    print(f"画像の保存形式は[{image_flag}]です。")

    model = YOLO("moucut_models/b6.pt")
    cnn_model = tf.keras.models.load_model("moucut_models/cnn.h5", compile=True)
    cap = cv2.VideoCapture(movie_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for_kmeans_array = []

    count = 0
    n = 0

    # Loop through the video frames
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            time.sleep(0.000005)
            pbar.update(1)
            n += 1
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame, device=device_name, verbose=False)
                try:
                    result = results[0].cpu().numpy()
                    ori_img = result.orig_img
                    box = result.boxes.xywh
                    # name = result.names
                    xcenter = box[0][0]
                    ycenter = box[0][1]
                    width = box[0][2]
                    height = box[0][3]

                    if width > height:
                        height = width
                    elif height > width:
                        width = height
                    left_top_x = math.floor(xcenter - (width / 2))
                    left_top_y = math.floor(ycenter - (height / 2))
                    right_btm_x = math.floor(xcenter + (width / 2))
                    right_btm_y = math.floor(ycenter + (height / 2))

                    croped = ori_img[left_top_y:right_btm_y, left_top_x:right_btm_x]
                    croped = cv2.resize(croped, (224, 224))

                    data = np.array(croped).astype(np.float32)
                    data = data[tf.newaxis]
                    x = tf.keras.applications.mobilenet_v3.preprocess_input(data)
                    cnn_result = cnn_model(x, training=False)
                    cnn_result = cnn_result.numpy()
                    cnn_result = cnn_result[0]

                    # print(cnn_result)

                except (IndexError, cv2.error):
                    # cnn_result = 0
                    pass

                if cnn_result[1] > 0.8:
                    for_kmeans_array.append(croped)
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot(line_width=(3))
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.rectangle(
                        annotated_frame, (0, 0), (50 + 800, 50 + 100), (80, 80, 80), -1
                    )
                    text_1 = f"CNN_score:OK[{cnn_result}]"
                    text_2 = f"Number of extractable images:{count}"
                    count += 1

                    cv2.putText(
                        annotated_frame,
                        text_1,
                        (50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )
                    cv2.putText(
                        annotated_frame,
                        text_2,
                        (50, 100),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )
                else:
                    annotated_frame = results[0].plot(line_width=(1))
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.rectangle(
                        annotated_frame, (0, 0), (50 + 800, 50 + 100), (80, 80, 80), -1
                    )
                    text_1 = "CNN_score:NG"
                    text_2 = f"Number of extractable images:{count}"

                    cv2.putText(
                        annotated_frame,
                        text_1,
                        (50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )
                    cv2.putText(
                        annotated_frame,
                        text_2,
                        (50, 100),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )

                # Display the annotated frame
                if show_flag is True:
                    cv2.imshow("Inference", annotated_frame)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # cv2.destroyWindow("YOLOv8 Inference")

    print("\033[32m顔検出完了\033[0m")

    print(f"検出数：[{count}]")

    cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
    movie_file_name = Path(movie_path).stem
    save_path = f"croped_image/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    kmeans.kmeans_main(
        save_path, movie_file_name, for_kmeans_array, cluster_num, image_flag
    )
    print("\033[32mAll Done!\033[0m")
