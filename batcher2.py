import os
import glob
import argparse
import shutil
from tqdm import tqdm
import platform
import subprocess

import getch

os_name = platform.system()


def masked_input(prompt=""):
    print(prompt, end="", flush=True)
    password = ""
    while True:
        char = getch.getch()
        if char == "\n" or char == "\r":  # Enterキーを検知
            break
        elif char == "\x08" or char == "\x7f":  # バックスペースを検知
            if len(password) > 0:
                # カーソルを戻してスペースで消して再度カーソルを戻す
                print("\b \b", end="", flush=True)
                password = password[:-1]
        else:
            print("*", end="", flush=True)
            password += char
    print()  # プロンプトを次の行に移動
    return password


def main(folder_path, num_items, pint, su_pass):

    # pass check
    max_attempts = 3
    attempts = 0

    if su_pass:
        while attempts < max_attempts:
            pass_input = masked_input("Please enter sudo password:")
            check_command = f"echo {pass_input} | sudo -S echo pass_check_ok"
            check_result = subprocess.run(
                check_command, shell=True, capture_output=True, text=True
            )
            attempts += 1
            if check_result.returncode == 0:
                print("sudo check ok")
                print("Output:", check_result.stdout)
                break
            else:
                print("password failed")
                # print("Error:", check_result.stderr)
                if attempts < max_attempts:
                    print("sudo パスワードが間違っています。もう一度試してください。")
                else:
                    return print("最大試行回数に達しました。終了します。")


    subdirectories = get_subdirectories(folder_path)
    for sub_dir in subdirectories:
        # print(sub_dir)
        exe_python = f"python batcher.py -f {sub_dir} -n {num_items} -p {pint} "
        os.system(exe_python)

        # メモリ開放用
        if su_pass:
            command = "sh clear_mem_cache.sh"
            result = subprocess.run(
                ["sudo", "-S"] + command.split(),
                input=str(pass_input) + "\n",
                text=True,
                capture_output=True,
            )
            print(result.stdout)
            print(result.stderr)

    print("all done")


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

    # opiton memoru purge pass
    parser.add_argument(
        "-ps",
        "--su_pass",
        action="store_true",
        help="バッチ処理中にメモリを開放したいときにSudo権限で開放するため",
        # type=str
        # required=True,
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    args = get_args()

    folder_path = args.folder_path
    num_items = args.num_items
    pint = args.pint
    su_pass = args.su_pass

    main(folder_path, num_items, pint, su_pass)
    print("\033[32m処理が完了しました。\033[0m")
