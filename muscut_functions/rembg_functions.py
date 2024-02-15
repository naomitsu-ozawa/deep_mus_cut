import multiprocessing

from rembg import new_session, remove
from tqdm import tqdm


def remove_bg(image, session, file_name):
    rembg_img = remove(image, session=session)
    return rembg_img, file_name


def process_rembg(images, imgnames, session):
    rembg_images = []
    file_names = []
    images = tqdm(images)

    for image, file_name in zip(images, imgnames):
        rembg_img, file_name = remove_bg(image, session, file_name)

        rembg_images.append(rembg_img)
        file_names.append(file_name)

    return rembg_images, file_names
