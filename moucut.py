import argparse
import os

from moucut_tools import (
    kmeans_image_extractor,
    moucut_default,
    webcam_list,
    moucut_sex_determination,
    moucut_sex_determination_yokogao,
    moucut_sex_determination_multi,
)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
):
    print("moucut.py_start")
    from ultralytics import YOLO

    if camera_list is True:
        mode = "camera_list"
        print("webcamera list")

        webcam_list.webcam_list()
        return
    if movie_path is None:
        print("-fオプションでファイルのパスかWebcamを指定して下さい。")
        return

    if device_flag is None:
        device_flag = "cpu"

    if image_flag is None:
        image_flag = "png"

    if tool is None:
        tool = "default"

    if mode is None:
        mode = "coreml"

    if mode == "coreml":
        import coremltools as ct

        yolo_model = YOLO("moucut_models/yolo.mlmodel", task="detect")
        cnn_model = ct.models.MLModel("moucut_models/ct_cnn.mlmodel")

    elif mode == "tf_pt":
        import tensorflow as tf

        yolo_model = YOLO("moucut_models/yolo.pt")
        cnn_model = tf.keras.models.load_model("moucut_models/cnn.h5", compile=True)

    elif mode == "trt_pt":
        import tensorflow as tf

        yolo_model = YOLO("moucut_models/yolo.pt")
        cnn_model = tf.saved_model.load("moucut_models/cnn_trt")

    if tool == "default":
        moucut_default.moucut(
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
        )

    elif tool == "kmeans_image_extractor":
        kmeans_image_extractor.main(movie_path, image_flag, cluster_num)

    elif tool == "tf2ml":
        from moucut_tools import tf2ml

        print("tools:tf2ml")
        tf2ml.tf2ml(movie_path)

    elif tool == "sexing":
        moucut_sex_determination.moucut(
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
        moucut_sex_determination_multi.moucut(
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
                cnn_model_2 = ct.models.MLModel("moucut_models/ct_cnn_2.mlmodel")
            except:
                print("横顔の雌雄判別モデルを[ct_cnn_2.mlmodel]として配置してください。")
        elif mode == "tf_pt":
            try:
                cnn_model_2 = tf.keras.models.load_model("moucut_models/ct_cnn_2.h5")
            except:
                print("横顔の雌雄判別モデルを[ct_cnn_2.h5]として配置してください。")

        elif mode == "trt_pt":
            try:
                cnn_model_2 = tf.saved_model.load("moucut_models/ct_cnn_2_trt")
            except:
                print("横顔の雌雄判別モデルを[ct_cnn_2.h5]として配置してください。")


        moucut_sex_determination_yokogao.moucut(
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
        )


def get_args():
    parser = argparse.ArgumentParser(
        prog="moucut_tool",  # プログラム名
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
    )
    print("\033[32m処理が完了しました。\033[0m")
