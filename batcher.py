import os
import glob
import argparse
import shutil
from tqdm import tqdm

import platform

os_name = platform.system()

def main(folder_path, num_items, pint):
    subdirectories = get_subdirectories(folder_path)
    for sub_dir in subdirectories:
        print(sub_dir)
        video_files = find_videos_in_subdirectories(sub_dir)
        # 動画ファイルの一覧を出力
        for video_file in video_files:
            print(video_file)

            exe_python = f"python muscut_with_rembg.py -f {video_file} -t extract_ok_frames -n {num_items} -p {pint}"
            os.system(exe_python)

        subdir_name = os.path.basename(sub_dir)
        destination_dir_name = f"{folder_path}/croped_imgs/{subdir_name}"
        create_directory_if_not_exists(destination_dir_name)
        image_files = glob.glob(f"./croped_image/**/**/*.png")
        move_image_files(image_files, destination_dir_name)

        del_dirs = get_subdirectories("./croped_image")

        for del_dir in del_dirs:
            shutil.rmtree(del_dir)
    print("all done")
    
    if os_name == "Linux":
        os.system("sh clear_mem_cache.sh")


def move_image_files(image_files, destination_folder):
    pbar = tqdm(image_files)
    for file_path in pbar:
        try:
            shutil.move(file_path, destination_folder)
            # print(f"ファイル {file_path} を {destination_folder} に移動しました")
        except Exception as e:
            print(f"ファイル {file_path} の移動中にエラーが発生しました: {e}")


def create_directory_if_not_exists(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"ディレクトリ {directory} を作成しました")
        else:
            print(f"ディレクトリ {directory} はすでに存在します")
    except Exception as e:
        print(f"ディレクトリ {directory} の作成中にエラーが発生しました: {e}")


def get_subdirectories(root_dir):
    subdirectories = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    return subdirectories


def find_videos_in_subdirectories(root_dir):
    video_extensions = ["MP4", "mp4", "avi", "mkv", "mov"]  # 動画ファイルの拡張子リスト
    videos = []  # 動画ファイルのリストを初期化

    for extension in video_extensions:
        search_pattern = (
            root_dir + "/**/*." + extension
        )  # ワイルドカードを使ってパスのパターンを作成
        videos.extend(
            glob.glob(search_pattern, recursive=True)
        )  # ワイルドカードを使って動画ファイルを検索し、リストに追加

    return videos


def get_args():
    parser = argparse.ArgumentParser(
        prog="muscut_batcher",  # プログラム名
        # usage="",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # opiton file name
    parser.add_argument(
        "-f",
        "--folder_path",
        help="解析したい動画のあるフォルダのパスを指定してください。",
        # type=str
        # required=True,
    )

    # option extract number
    parser.add_argument(
        "-n",
        "--num_items",
        type=int,
        help="抽出枚数を指定してください",
    )

    # option pint check
    parser.add_argument(
        "-p",
        "--pint",
        type=int,
        help="ピントチェックの閾値、デフォルト2600",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    args = get_args()

    folder_path = args.folder_path
    num_items = args.num_items
    pint = args.pint

    main(folder_path, num_items, pint)
    print("\033[32m処理が完了しました。\033[0m")


##############test##################

# folder_path = (
#     "/media/idm-kurume/ex_ssd/ex_ssd/project_MYZ/formjudge/sex_test/formjudge_test"
# )

# num_items = 30

# main(folder_path, num_items)

############ テスト用#################
