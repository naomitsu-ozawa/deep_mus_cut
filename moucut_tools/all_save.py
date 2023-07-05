# %%
import os
from pathlib import Path

import cv2
from tqdm import tqdm


# %%
def main(movie_path, img_array, image_flag):
    if movie_path == "webcam":
        movie_path = 0
        movie_file_name = "webcam"
    else:
        movie_path = movie_path
        movie_file_name = Path(movie_path).stem

    save_path = f"croped_image/{movie_file_name}/all_extract_image"
    os.makedirs(save_path, exist_ok=True)

    pbar = tqdm(img_array)
    count = 0
    for item in pbar:
        count_number = str(count).zfill(6)
        if image_flag == "jpg":
            cv2.imwrite(save_path + "/extract_{}.jpg".format(count_number), item)
        else:
            cv2.imwrite(save_path + "/extract_{}.png".format(count_number), item)
        count += 1
