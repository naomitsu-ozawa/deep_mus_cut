# %%
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from multiprocessing import Pool


# %%
def save_image(args):
    movie_file_name, frame_number, image_flag, save_path, item = args
    if image_flag == "jpg":
        cv2.imwrite(
            save_path + "/extract_{}_{}.jpg".format(movie_file_name, frame_number), item
        )
    elif image_flag == "png":
        cv2.imwrite(
            save_path + "/extract_{}_{}.png".format(movie_file_name, frame_number), item
        )


def main(movie_path, img_array, for_kmeans_frame_no, image_flag):
    if movie_path == "webcam":
        movie_path = 0
        movie_file_name = "webcam"
    else:
        movie_path = movie_path
        movie_file_name = Path(movie_path).stem

    save_path = f"croped_image/{movie_file_name}/selected_imgs"
    os.makedirs(save_path, exist_ok=True)

    items = zip(for_kmeans_frame_no, img_array)

    with Pool() as pool:
        args = [
            (movie_file_name, f"frame_{item_name}", image_flag, save_path, item)
            for item_name, item in items
        ]
        pbar = tqdm(pool.imap_unordered(save_image, args), total=len(args))
        for _ in pbar:
            pass

    # pbar = tqdm(items)

    # for item_name, item in pbar:
    #     # count_number = str(count).zfill(6)
    #     frame_number = f"frame_{item_name}"
    #     if image_flag == "jpg":
    #         cv2.imwrite(save_path + "/extract_{}_{}.jpg".format(movie_file_name, frame_number), item)
    #     else:
    #         cv2.imwrite(save_path + "/extract_{}_{}.png".format(movie_file_name, frame_number), item)
