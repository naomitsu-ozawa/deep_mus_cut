# Import Lib
import gc
import glob
import math
import os
import platform
import random

# import warnings
from pathlib import Path
from time import sleep, time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

from muscut_functions import rembg_functions, global_functions, cv_functions
import multiprocessing

os_name = platform.system()


def main(input_path):

    imgae_path = f"{input_path}/selected_imgs"
    output_folder = f"{input_path}/rembg_imgs"
    imgs_files = glob.glob(f"{imgae_path}/*.png")

    global_functions.create_directory_if_not_exists(output_folder)

    imgaes, imgnames = cv_functions.read_images_parallel(imgs_files)

    if os_name == "Darwin":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # rembgで使うマスク抽出モデル
    # u2net default
    # u2netp light weight model
    # isnet-general-use new pre trained model
    # isnet-anime
    # sam
    # silueta

    print("\033[32m背景除去開始\033[0m")
    unet_model_name = "isnet-general-use"
    print(f"model:{unet_model_name}")
    session = new_session(
        unet_model_name,
        providers=providers,
    )

    rembg_images, output_paths = rembg_functions.process_rembg(
        imgaes, imgnames, session, output_folder
    )

    datas = tqdm(zip(rembg_images, output_paths), desc="Saving...")
    for img_data, path in datas:
        # 画像を保存する
        cv2.imwrite(path, img_data)

    print("\033[32m背景除去完了\033[0m")
