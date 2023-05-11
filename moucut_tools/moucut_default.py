# %%
import datetime
import math
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from moucut_tools import kmeans


# %%
def moucut(movie_path, device, image_flag, show_flag, yolo_model, cnn_model, mode, cluster_num, wc_flag):
    if movie_path == "webcam":
        timezone = datetime.timezone(datetime.timedelta(hours=+9), "JST")
        dt = datetime.datetime.now()
        dt = dt.astimezone(timezone)
        dt = "{0:%Y-%m-%d-%H-%M-%S}".format(dt)
        movie_path = 0
        movie_file_name = f"webcam{dt}"
    else:
        movie_file_name = Path(movie_path).stem

    if mode == "coreml":
        # import coremltools as ct
        running_mode = "CoreML"
    elif mode == "tf":
        import tensorflow as tf
        running_mode = "TensorFlow&PyTorch"

    os_name = platform.system()

    print(f"OS:{os_name}")
    print(f"runing mode :{running_mode}")
    print(f"image format:[{image_flag}]です。")

    if os_name == "Darwin":
        cap = cv2.VideoCapture(movie_path, cv2.CAP_AVFOUNDATION)
    else:
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
            cnn_result = 0
            if success:
                # Run YOLOv8 inference on the frame
                results = yolo_model(frame, verbose=False)
                try:
                    if mode == "coreml":
                        result = results[0].numpy()
                    elif mode == "tf":
                        result = results[0].cpu().numpy()
                    else:
                        print("modeを指定して下さい")
                        return

                    ori_img = result.orig_img
                    length = result.boxes.shape[0]
                    for i in range(length):
                        # print('roop!')
                        box = result[i].boxes.xywh
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

                        if not wc_flag:
                            if mode == "coreml":
                                img_np = np.array(croped).astype(np.float32)
                                img_np = img_np[np.newaxis, :, :, :]
                                cnn_result = cnn_model.predict({"input_1": img_np})
                                cnn_result = cnn_result["Identity"][0][1]
                                if cnn_result > 0.8:
                                    for_kmeans_array.append(croped)
                                    count += 1

                            elif mode == "tf":
                                data = np.array(croped).astype(np.float32)
                                data = data[tf.newaxis]
                                x = tf.keras.applications.mobilenet_v3.preprocess_input(data)
                                cnn_result = cnn_model(x, training=False)
                                cnn_result = cnn_result.numpy()
                                cnn_result = cnn_result[0]
                                if cnn_result[1] > 0.8:
                                    for_kmeans_array.append(croped)
                                    count += 1
                        else:
                            cnn_result = "without cnn"
                            for_kmeans_array.append(croped)
                            count += 1

                except (IndexError, cv2.error):
                    cnn_result = 0
                    pass

                if show_flag is True:
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot(line_width=(3))
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.rectangle(annotated_frame, (0, 0), (360, 50), (80, 80, 80), -1)
                    text_1 = f"CNN_score:[{cnn_result}]"
                    text_2 = f"Number of extractable images:{count}"

                    cv2.putText(
                        annotated_frame,
                        text_1,
                        (5, 17),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(250, 250, 250),
                    )
                    cv2.putText(
                        annotated_frame,
                        text_2,
                        (5, 37),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(250, 250, 250),
                    )

                    # Display the annotated frame
                    cv2.imshow("Inference", annotated_frame)
                    key = cv2.waitKey(1)
                    if key == 27:
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

    if cluster_num is None:
        cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
    else:
        pass

    save_path = f"croped_image/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    kmeans.kmeans_main(
        save_path, movie_file_name, for_kmeans_array, cluster_num, image_flag
    )
    print("\033[32mAll Done!\033[0m")
