import argparse
import logging
import math
import os
import platform
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from muscut_functions import cv_functions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os_name = platform.system()


class NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


# 既存のハンドラをクリアして、再設定
for handler in LOGGER.handlers:
    LOGGER.removeHandler(handler)

# 新しいハンドラで INFO 以上を通しつつ WARNING を除外
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.addFilter(NoWarningFilter())

LOGGER.addHandler(console_handler)
LOGGER.setLevel(logging.INFO)


def focus_value(image):
    p = cv2.Sobel(image, dx=1, dy=1, ddepth=cv2.CV_8U, ksize=5, scale=1, delta=50).var()
    return p


import math

import matplotlib.pyplot as plt


def show_images_in_grid(
    images,
    scores,
    columns=5,
    image_size_px=(300, 300),
    title=None,
    title_fontsize=14,
    score_fontsize=12,
    dpi=100,
):
    """
    images: 画像（Tensor）リスト
    scores: 各画像に対応する数値（例：分散値）
    columns: 1行に並べる画像数
    image_size_px: 各画像の表示サイズ（ピクセル） tuple: (width, height)
    title: 図の上部タイトル（長文可）
    title_fontsize: タイトルの文字サイズ
    score_fontsize: 各画像下のスコア表示の文字サイズ
    dpi: 1インチあたりのピクセル密度（デフォルト100）
    """
    rows = math.ceil(len(images) / columns)

    fig_width = columns * image_size_px[0] / dpi
    fig_height = rows * image_size_px[1] / dpi

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    if title:
        # y を上げて、タイトルが画像と被らないように
        plt.suptitle(title, fontsize=title_fontsize, y=1)

    for i, (img, score) in enumerate(zip(images, scores)):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(img.numpy().astype(np.uint8))
        plt.axis("off")
        plt.title(f"{score:.1f}", fontsize=score_fontsize)

    # タイトルが切れないように、上の余白を広めに
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def main(movie, model, cnn_model, num_images=10, b_size=16):
    if not os.path.exists(movie):
        print(f"Movie file {movie} does not exist.")
        return

    print("Starting Head Detection...")
    start_time = time.time()
    results = model.predict(
        source=movie,
        stream=False,
        conf=0.5,
        verbose=True,
        batch=b_size,
    )
    end_time = time.time()
    print(f"Head detection completed in {end_time - start_time:.2f} seconds.")

    print(f"Number of head detections: {len(results)}")

    head_images = []

    # tqdmを使って進捗バーを表示
    for result in tqdm(results, desc="Processing Crps", unit="image"):
        # for result in results:
        if result.boxes:
            full_image = result.orig_img[..., ::-1]  # Convert BGR to RGB
            (
                left_top_x,
                left_top_y,
                right_btm_x,
                right_btm_y,
                cv_top_x,
                cv_top_y,
                cv_btm_x,
                cv_btm_y,
            ) = cv_functions.crop_modified_xy(result)
            head_image = full_image[left_top_y:right_btm_y, left_top_x:right_btm_x]
            head_images.append(head_image)

    input_size = (224, 224)  # 高さ, 幅（注意：cv2と2は順番逆）

    processed_images = []
    for img in head_images:
        if img is None or img.size == 0:
            continue  # 空の画像はスキップ

        img_np = img.astype(np.float32)  # 必ず float32 に
        img_tf = tf.convert_to_tensor(img_np)  # NumPy → Tensor

        # TensorFlowでリサイズ
        img_resized = tf.image.resize(img_tf, input_size)  # (224, 224, 3)
        processed_images.append(img_resized)

    # バッチ化（Tensorのリスト → Tensor）
    input_tensor = tf.stack(processed_images)  # shape: (N, 224, 224, 3)
    # 推論
    print("Starting Side-Profile Classification...")
    cnn_results = cnn_model.predict(input_tensor, batch_size=b_size, verbose=1)
    print(f"Number of side-profile classifications: {len(cnn_results)}")

    # 各予測で最大スコアのインデックスを取得
    predicted_classes = np.argmax(cnn_results, axis=1)
    # print(predicted_classes)

    # クラス1だけ抽出
    result_images = [
        img for img, cls in zip(processed_images, predicted_classes) if cls == 1
    ]

    # すべての画像の分散値（フォーカススコア）を計算
    print("Calculating Focus Scores...")
    focus_scores = [focus_value(img.numpy()) for img in result_images]

    # 分散値の範囲（最小〜最大）を取得し、10段階で等間隔分割
    min_score = min(focus_scores)
    max_score = max(focus_scores)
    thresholds = np.linspace(min_score, max_score, num_images)

    print("min_score:", min_score)
    print("max_score:", max_score)
    print("thresholds:", thresholds)

    # ステップ2：各 threshold に対して最も近いスコアを持つ画像を1枚選ぶ
    selected_images = []
    selected_scores = []

    for t in thresholds:
        # しきい値以下のスコアを抽出
        candidates = [
            (img, score)
            for img, score in zip(result_images, focus_scores)
            if score <= t
        ]

        if not candidates:
            continue  # しきい値以下の画像がない場合はスキップ

        # 最も t に近いスコアの画像を選ぶ（score - t が最小）
        closest_img, closest_score = min(candidates, key=lambda x: abs(x[1] - t))

        selected_images.append(closest_img)
        selected_scores.append(closest_score)

    show_images_in_grid(
        images=selected_images,
        scores=selected_scores,
        columns=5,  # 任意に指定（3〜6あたりが見やすい）
        image_size_px=(224, 224),  # 1画像あたりのサイズを維持（ピクセル単位）
        title=(
            f"These {num_images} side-profile images were extracted from a video and ranked by focus score (variance) across {num_images} levels. \n Use them as a visual reference for evaluating sharpness. \nBased on the comparison between images and their displayed focus scores, \nplease set a threshold above the value of noticeably blurry images."
        ),
    )


def get_args():
    # option file name
    parser = argparse.ArgumentParser(
        description="Focus value extraction from video or webcam using YOLO and CNN models."
    )
    parser.add_argument(
        "-f",
        "--movie_path",
        help="ファイルのパスかwebcamを指定して下さい。['file_path','movie_path','webcam']",
        # required=True,
    )

    # option extract number
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="抽出枚数",
    )

    # option batch size
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="バッチサイズ",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    # 引数の取得
    args = get_args()
    movie = args.movie_path
    num_images = args.number
    b_size = args.batch_size

    print(f"Starting video processing: {movie}")
    print(f"Number of images to extract: {num_images}")

    print("Loading models...")
    if os_name == "Darwin":
        model = YOLO("muscut_models/yolo.mlmodel", task="detect")
    elif os_name == "Linux" or os_name == "Windows":
        model = YOLO("muscut_models/yolo.pt")

    cnn_model = tf.keras.models.load_model("muscut_models/cnn/savedmodel")
    print("Models loaded successfully.")

    main(movie, model, cnn_model, num_images, b_size)

    print("Focus value extraction completed.")
