import argparse
import logging
import math
import os
import platform
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec as mgs
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from muscut_functions import cv_functions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os_name = platform.system()

import tensorflow as tf


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


def plot_images_above_histogram_with_thresholds(
    focus_scores,
    selected_images,
    selected_scores,
    thresholds,
    columns=5,
    image_size_px=(224, 224),
    dpi=96,
    spacing_px=5,
    hist_height_px=300,
    bins=100,
):
    rows = math.ceil(len(selected_images) / columns)
    img_w, img_h = image_size_px

    grid_w_px = columns * img_w + (columns - 1) * spacing_px
    grid_h_px = rows * img_h + (rows - 1) * spacing_px

    fig_w_in = grid_w_px / dpi
    fig_h_in = (grid_h_px + hist_height_px) / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=True)

    gs = mgs.GridSpec(2, 1, height_ratios=[grid_h_px, hist_height_px], figure=fig)

    # === タイトル ===
    fig.text(
        0.5,
        0.98,
        f"Focus Score Histogram with {len(selected_images)} Representative Images",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
    )

    # === 画像グリッド ===
    grid_gs = mgs.GridSpecFromSubplotSpec(
        rows,
        columns,
        subplot_spec=gs[0],
        wspace=spacing_px / img_w,
        hspace=spacing_px / img_h,
    )

    for i, (img, score) in enumerate(zip(selected_images, selected_scores)):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(grid_gs[row, col])
        ax.imshow(img.numpy().astype(np.uint8))
        ax.axis("off")
        ax.text(
            0.5,
            0.15,
            f"{score:.1f}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="black", edgecolor="none", boxstyle="round,pad=0.2"),
        )

    # === ヒストグラム ===
    ax_hist = fig.add_subplot(gs[1])
    sns.histplot(
        focus_scores,
        bins=bins,
        color="#c04e01",
        kde=True,
        ax=ax_hist,
        alpha=0.4,
        linewidth=0,
    )

    counts_per_bin = []
    for i in range(len(thresholds) - 1):
        if i < len(thresholds) - 2:
            cond = (np.array(focus_scores) >= thresholds[i]) & (
                np.array(focus_scores) < thresholds[i + 1]
            )
        else:
            cond = (np.array(focus_scores) >= thresholds[i]) & (
                np.array(focus_scores) <= thresholds[i + 1]
            )
        counts_per_bin.append(np.sum(cond))

    # === 赤線区間で背景色をつける ===
    sorted_scores = sorted(selected_scores)
    counts = []

    # スコア間の画像枚数を数える
    for i in range(len(sorted_scores) - 1):
        t_start = sorted_scores[i]
        t_end = sorted_scores[i + 1]
        count = np.sum(
            (np.array(focus_scores) >= t_start) & (np.array(focus_scores) < t_end)
        )
        counts.append(count)

    max_count = max(counts) if counts else 1  # ゼロ除算対策

    # 色の帯を描画（赤線間の領域）
    for i in range(len(counts)):
        t_start = sorted_scores[i]
        t_end = sorted_scores[i + 1]
        alpha = min(counts[i] / max_count * 0.5, 0.5)
        ax_hist.axvspan(t_start, t_end, alpha=alpha, color="teal", zorder=0)
        ax_hist.text(
            (t_start + t_end) / 2,
            ax_hist.get_ylim()[1] * 0.92,
            f"{counts[i]} imgs",
            ha="center",
            fontsize=8,
            color="black",
        )

    for score in selected_scores:
        ax_hist.axvline(score, color="red", linestyle="--", alpha=0.8)
        ax_hist.text(
            score,
            ax_hist.get_ylim()[1] * 0.75,
            f"{score:.1f}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="red",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.75,
            ),
        )

    ax_hist.set_title("Focus Score Distribution with Thresholds", fontsize=13)
    ax_hist.set_xlabel("Focus Score (Variance)")
    ax_hist.set_ylabel("Frequency")

    plt.show()


def main(movie, model, cnn_model, num_images=10, b_size=8):

    global_start_time = time.time()

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

    global_end_time = time.time()
    print(f"Total processing time: {global_end_time - global_start_time:.2f} seconds")

    plot_images_above_histogram_with_thresholds(
        focus_scores=focus_scores,
        selected_images=selected_images,
        selected_scores=selected_scores,
        thresholds=thresholds,
        bins=int(len(cnn_results) / 10),
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
