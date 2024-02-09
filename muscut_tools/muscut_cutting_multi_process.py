from muscut_functions import cv_functions, global_functions, cutting_functions
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm


def main(input_path, device, yolo_model, mode):
    print("切り取り中・・・")

    imgs_files = glob.glob(f"{input_path}/rembg_imgs/*.png")
    images, imgnames = cv_functions.read_images_parallel(imgs_files)

    output_folder = f"{input_path}/with_rembg"
    global_functions.create_directory_if_not_exists(output_folder)

    croped_imgs, output_paths = cutting_functions.process_cutting(
        images, device, yolo_model, mode, output_folder, imgnames
    )
    print("切り取り完了")
    datas = tqdm(zip(croped_imgs, output_paths))
    for img_data, path in datas:
        # 画像を保存する
        cv2.imwrite(path, img_data)
