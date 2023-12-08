import argparse
import os
import platform

from muscut_tools import (
    kmeans_image_extractor,
    muscut_default,
    webcam_list,
    muscut_sex_determination,
    muscut_sex_determination_yokogao,
    muscut_sex_determination_multi,
)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os_name = platform.system()


def main(
    movie_path,
    device_flag,
    image_flag,
    tool,
    show_flag,
    cluster_num,
    mode,
    wc_flag,
    camera_list,
    all_extract,
    cnn_conf,
):
    print("muscut.py_start")
    from ultralytics import YOLO

    if camera_list is True:
        mode = "camera_list"
        print("webcamera list")

        webcam_list.webcam_list()
        return
    if movie_path is None:
        print("-fオプションでファイルのパスかWebcamを指定して下さい。")
        return

    # if device_flag is None:
    #     device_flag = "cpu"

    if image_flag is None:
        image_flag = "png"

    if tool is None:
        tool = "default"

    if mode is None:
        if os_name == "Darwin":
            mode = "coreml"
        elif os_name == "Linux" or os_name == "Windows":
            mode = "tf_pt"

    if mode == "coreml":
        import coremltools as ct

        yolo_model = YOLO("muscut_models/yolo.mlmodel", task="detect")
        cnn_model = ct.models.MLModel("muscut_models/ct_cnn.mlmodel")

    elif mode == "tf_pt":
        import tensorflow as tf

        GPU_flag = tf.test.is_gpu_available()
        if GPU_flag and device_flag is None:
            if os_name == "Darwin":
                device_flag = "mps"
                print("GPU: metal")
            else:
                device_flag = "cuda"
                print("GPU: CUDA")
        else:
            device_flag = "cpu"
            print("GPU: not use")

        yolo_model = YOLO("muscut_models/yolo.pt")
        cnn_model = tf.saved_model.load("muscut_models/cnn/savedmodel")

    if cnn_conf is None:
        cnn_conf = 0.9
        print("cnn_conf: default 0.9")
    else:
        print(f"cnn_conf = {cnn_conf}")

    if tool == "default":
        muscut_default.muscut(
            movie_path,
            device_flag,
            image_flag,
            show_flag,
            yolo_model,
            cnn_model,
            mode,
            cluster_num,
            wc_flag,
            all_extract,
            cnn_conf,
        )

    elif tool == "kmeans_image_extractor":
        kmeans_image_extractor.main(movie_path, image_flag, cluster_num)

    elif tool == "tf2ml":
        from muscut_tools import tf2ml

        print("tools:tf2ml")
        tf2ml.tf2ml(movie_path)

    elif tool == "sexing":
        muscut_sex_determination.muscut(
            movie_path,
            device_flag,
            image_flag,
            show_flag,
            yolo_model,
            cnn_model,
            mode,
            cluster_num,
            wc_flag,
        )

    elif tool == "sexing_multi":
        muscut_sex_determination_multi.muscut(
            movie_path,
            device_flag,
            image_flag,
            show_flag,
            yolo_model,
            cnn_model,
            mode,
            cluster_num,
            wc_flag,
        )

    elif tool == "sexing_yokogao":
        if mode == "coreml":
            try:
                cnn_model_2 = ct.models.MLModel("muscut_models/ct_cnn_2.mlmodel")
            except:
                print("横顔の雌雄判別モデルを[ct_cnn_2.mlmodel]として配置してください。")
        elif mode == "tf_pt":
            try:
                cnn_model_2 = tf.saved_model.load("muscut_models/ct_cnn_2/savedmodel")
            except:
                print("横顔の雌雄判別モデルを[ct_cnn_2/savedmodel]として配置してください。")

        muscut_sex_determination_yokogao.muscut(
            movie_path,
            device_flag,
            image_flag,
            show_flag,
            yolo_model,
            cnn_model,
            cnn_model_2,
            mode,
            cluster_num,
            wc_flag,
            cnn_conf
        )


def get_args():
    parser = argparse.ArgumentParser(
        prog="muscut_tool",  # プログラム名
        # usage="",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    # opiton file name
    parser.add_argument(
        "-f",
        "--movie_path",
        help="ファイルのパスかwebcamを指定して下さい。['file_path','movie_path','webcam']",
        # required=True,
    )

    # option mode
    parser.add_argument(
        "-m",
        "--mode",
        help="tensorflow&pytorchを利用する or coremlを利用する['tf_pt','coreml']",
        type=str,
    )

    # option tf&pt device option
    parser.add_argument(
        "-d",
        "--device_flag",
        help="mode=tf_ptで利用するデバイスを指定して下さい。指定しない場合は、['cpu']で実行します。['cpu','cuda']",
        type=str,
    )

    # option tools
    parser.add_argument(
        "-t",
        "--tool",
        help="使用するツールを指定して下さい。['default','kmeans_image_extractor']",
        type=str,
    )

    # option image format
    parser.add_argument(
        "-i",
        "--image_format",
        help="画像の保存フォーマットを指定して下さい。指定しない場合は、['png']で保存します。['jpg','png']",
    )

    # option preview show
    parser.add_argument(
        "-s",
        "--show_flag",
        action="store_true",
        help="検知状況を表示します[True or False]",
    )

    parser.add_argument("-c", "--cnn_conf", type=float, help="画像分類モデルの閾値を、少数で設定")

    # option preview show
    parser.add_argument(
        "-wc",
        "--without_cnn",
        action="store_true",
        help="CNNモデルによる分類をスキップします。",
    )

    # option extract number
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="抽出枚数",
    )

    # option webcamera list
    parser.add_argument(
        "-cl",
        "--camera_list",
        action="store_true",
        help="webcamera list",
    )

    # option all extract
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="all_extract",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    args = get_args()

    movie_path = args.movie_path
    device_name = args.device_flag
    image_format = args.image_format
    tool = args.tool
    show_flag = args.show_flag
    cluster_num = args.number
    mode = args.mode
    wc_flag = args.without_cnn
    camera_list = args.camera_list
    all_extract = args.all
    cnn_conf = args.cnn_conf

    main(
        movie_path,
        device_name,
        image_format,
        tool,
        show_flag,
        cluster_num,
        mode,
        wc_flag,
        camera_list,
        all_extract,
        cnn_conf,
    )
    print("\033[32m処理が完了しました。\033[0m")
