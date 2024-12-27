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

import onnxruntime as ort

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
    imgs_files = glob.glob(f"{imgae_path}/*.png")

    images, imgnames = cv_functions.read_images_parallel(imgs_files)


    session_options = ort.SessionOptions()

    cuda_provider_options = {
        # 'gpu_mem_linit': '2147483648'
        # 'gpu_mem_linit': '1610612736'
        'gpu_mem_linit': 1.4 * 1024 * 1024 * 1024
        # 'gpu_mem_linit': 0.5 * 1024 * 1024 * 1024
    }

    if os_name == "Darwin":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        # providers = [("CUDAExecutionProvider", "CPUExecutionProvider", cuda_provider_options)]
        providers = ["CUDAExecutionProvider", cuda_provider_options]

    # rembgで使うマスク抽出モデル
    # u2net default
    # u2netp light weight model
    # isnet-general-use new pre trained model
    # isnet-anime
    # sam
    # silueta

    print("\033[32m背景除去開始\033[0m")
    unet_model_name = "isnet-general-use"
    print(f"\033[32m start rembg model:{unet_model_name}\033[0m")

    session = new_session(
        unet_model_name,
        sess_options=session_options,
        providers=providers,
    )

    # Check if the session is using GPU
    if 'CUDAExecutionProvider' in session.providers:
        print("\033[32mONNX Runtime is using GPU.\033[0m")
    else:
        print("\033[31mONNX Runtime is not using GPU.\033[0m")

    rembg_images, file_names = rembg_functions.process_rembg(
        images, imgnames, session
    )

    print("\033[32m背景除去完了\033[0m")

    return rembg_images, file_names
