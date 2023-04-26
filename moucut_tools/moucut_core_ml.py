import math
import os
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
from tqdm import tqdm

from moucut_tools import kmeans

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def moucut(movie_path, device_flag, image_flag):
    print("runing_CoreML_version")
    print(f"物体検出に{device_flag}を利用します。")
    print(f"画像の保存形式は[{image_flag}]です。")
    cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
    # model path
    model_path = "moucut_models/yolo.pt"
    cnn_model_path = "moucut_models/ct_cnn.mlmodel"
    # source video path
    movie_file = movie_path
    movie_file_name = Path(movie_file).stem
    # モデルの読み込み
    cnn_model = ct.models.MLModel(cnn_model_path)
    print("\033[32mCNN_mobile_netv3モデル読み込み完了\033[0m")
    # "cuda" "cpu" "mps" mps is mac m1 apple silicon gpu
    device_type = torch.device(device_flag)  # pylint: disable=no-member
    model = torch.hub.load(".", "custom", path=model_path, source="local")
    model.to(device_type)
    print("\033[32myolov5モデルの読み込み完了\033[0m")
    # 保存先
    save_path = f"croped_image/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    for_kmeans_array = []

    print("\033[32m\033[1m検出開始\033[0m")
    print("\033[32m動画から顔を検出中・・・\033[0m")

    cv2movie = cv2.VideoCapture(movie_file)
    nframe = int(cv2movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = range(nframe)

    for frame in tqdm(frames):
        ret, frame = cv2movie.read()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame

        model.conf = 0.8
        # 物体検出
        with torch.no_grad():
            results = model(img)
        # 予測結果の出力からキーワードを指定して条件分岐させる
        res_str = str(results)
        res_flag = "no detections" in res_str  # 未検出
        if res_flag:
            pass
        else:
            try:
                coordinates = results.pandas().xywh[0]
                xcenter = coordinates.xcenter[0]
                ycenter = coordinates.ycenter[0]
                width = coordinates.width[0]
                height = coordinates.height[0]
                if width > height:
                    height = width
                elif height > width:
                    width = height
                left_top_x = math.floor(xcenter - (width / 2))
                left_top_y = math.floor(ycenter - (height / 2))
                right_btm_x = math.floor(xcenter + (width / 2))
                right_btm_y = math.floor(ycenter + (height / 2))

                croped = img[left_top_y:right_btm_y, left_top_x:right_btm_x]
                croped = cv2.resize(croped, (224, 224))
                img_np = np.array(croped).astype(np.float32)
                img_np = img_np[np.newaxis, :, :, :]

                cnn_result = cnn_model.predict({"input_1": img_np})
                cnn_result = cnn_result["Identity"][0][1]

                if cnn_result > 0.8:
                    for_kmeans_array.append(croped)
            except cv2.error:
                pass

    print("\033[32m顔検出完了\033[0m")
    kmeans.kmeans_main(
        save_path, movie_file_name, for_kmeans_array, cluster_num, image_flag
    )
    print("\033[32mAll Done!\033[0m")