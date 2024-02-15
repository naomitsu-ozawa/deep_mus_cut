from muscut_functions import cv_functions, global_functions, cutting_functions
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm


def main(input_path, rembg_images, image_names, device, yolo_model, mode):
    print("\033[32m切り取り中・・・\033[0m")

    output_folder = f"{input_path}/with_rembg"
    global_functions.create_directory_if_not_exists(output_folder)

    croped_imgs, output_paths = cutting_functions.process_cutting(
        rembg_images, device, yolo_model, mode, output_folder, image_names
    )
    print("\033[32m切り取り完了\033[0m")
    datas = tqdm(zip(croped_imgs, output_paths), desc="Saving...")
    for img_data, path in datas:
        # 画像を保存する
        cv2.imwrite(path, img_data)
