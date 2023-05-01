import argparse
import os

# from moucut_tools import moucut_tf,moucut_core_ml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(file, device, image, tool, show):
    if device is None:
        device = "cpu"
    if image is None:
        image = "png"

    if tool is None:
        tool = "tf"

    if tool == "tf":
        from moucut_tools import moucut_tf

        moucut_tf.moucut(file, device, image, show)
    elif tool == "coreml":
        device = "mps"
        from moucut_tools import moucut_coreml

        moucut_coreml.moucut(file, device, image, show)
    elif tool == "kmeans_image_extractor":
        from moucut_tools import kmeans_image_extractor

        kmeans_image_extractor.main(file, image)
    elif tool == "all_extract":
        from moucut_tools import all_extract

        all_extract.moucut(file, device, image, show)
    elif tool == "all_extract_test":
        from moucut_tools import all_extract_test

        all_extract_test.moucut(file, device, image, show)


def get_args():
    parser = argparse.ArgumentParser(
        prog="moucut_tool",  # プログラム名
        # usage="",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    parser.add_argument("-f", "--file", help="動画ファイルのパスを指定して下さい。['file_path','webcam']", required=True)

    parser.add_argument(
        "-d",
        "--device",
        help="物体検知で利用するデバイスを指定して下さい。指定しない場合は、['cpu']で実行します。['cpu','cuda','mps']",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--tool",
        help="使用するツールを指定して下さい。['tf','coreml','kmeans_image_extractor','all_extract']\ntf => 画像分類にTensorflowを使います。",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--image_format",
        help="出力する画像のフォーマットを指定して下さい。指定しない場合は、['png']で保存します。['jpg','png']",
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="検知状況を表示します[True or False]",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    args = get_args()

    file_path = args.file
    device_name = args.device
    image_format = args.image_format
    tool = args.tool
    show = args.show

    main(file_path, device_name, image_format, tool, show)
    print("\033[32m処理が完了しました。\033[0m")
