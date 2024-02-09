import multiprocessing

from rembg import new_session, remove
from tqdm import tqdm


def remove_bg(image, session, file_name, output_folder):
    rembg_img = remove(image, session=session)
    output_path = f"{output_folder}/{file_name}_rembgout.png"
    return rembg_img, output_path


def process_rembg(images, imgnames, session, output_folder):
    rembg_images = []
    output_paths = []
    images = tqdm(images)
    for image, file_name in zip(images, imgnames):
        rembg_img, output_path = remove_bg(image, session, file_name, output_folder)

        rembg_images.append(rembg_img)
        output_paths.append(output_path)
    return rembg_images, output_paths
