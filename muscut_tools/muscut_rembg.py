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

os_name = platform.system()

if os_name == "Darwin":
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def rembg_it(files, output_path, session):
    files = tqdm(files)
    for file in files:
        file_name = os.path.split(file)[1]
        input_path = str(file)
        save_image_path = f"{output_path}/{file_name}_rembgout.png"

        input = cv2.imread(input_path)
        output = remove(input, session=session)
        cv2.imwrite(save_image_path, output)
        sleep(0.0000002)
        files.set_description("Processing %s" % file_name)


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(save_path):
    # u2net model
    # u2net default
    # u2netp light weight model
    # isnet-general-use new pre trained model
    # isnet-anime
    # sam
    # silueta
    print("背景除去開始")
    unet_model_name = "isnet-general-use"
    print(f"model:{unet_model_name}")
    session = new_session(
        unet_model_name,
        # providers=['TensorrtExecutionProvider','CUDAExecutionProvider']
        providers=providers,
        # providers=['CPUExecutionProvider']
        # providers=['TensorrtExecutionProvider']
    )

    files_path = f"{save_path}/selected_imgs"
    print(files_path)
    files = glob.glob(f"{files_path}/*.png")
    # print(files)

    output_path = f"{save_path}/rembg_imgs"
    create_directory_if_not_exists(output_path)

    rembg_it(files, output_path, session)

    print("背景除去完了")
