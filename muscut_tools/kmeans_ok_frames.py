import glob
import os
import random
import re
import shutil

import cv2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import multiprocessing


import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


# kmeansのモデル構築
def build_kmeans(df, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, n_init="auto")
    kmeans.fit(df)
    return kmeans


# 主成分分析のモデル構築
def build_pca(df):
    pca = PCA()
    pca.fit(df)
    return pca


###########################################################################3
def process_images(
    start,
    end,
    imgs_list,
    save_path,
    kmeans,
    format_flag,
    video_name,
    for_kmeans_frame_no,
    progress_queue,
):
    for j in range(start, end):
        label = kmeans.labels_[j]
        img = imgs_list[j]
        # file_number = str(frame_no).zfill(6)
        file_number = for_kmeans_frame_no[j].zfill(6)
        if format_flag == "jpg":
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(label, f"{video_name}-{file_number}.jpg"),
                img,
            )
        else:
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(label, f"{video_name}-{file_number}.png"),
                img,
            )
    progress_queue.put(1)  # プロセスの処理完了をキューに追加


def make_cluster_dir_parallel(
    imgs_list, save_path, kmeans, format_flag, video_name, for_kmeans_frame_no
):
    # 保存先のディレクトリを空にして作成
    shutil.rmtree(save_path)
    os.mkdir(save_path)
    # クラスタごとのディレクトリ作成
    for i in tqdm(range(kmeans.n_clusters), desc="Creating directories"):
        cluster_dir = save_path + "cluster{}".format(i)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

    num_processes = multiprocessing.cpu_count()
    processes = []
    imgs_per_process = len(imgs_list) // num_processes
    progress_queue = multiprocessing.Queue()  # プロセスの進捗を管理するためのキュー

    for i in range(num_processes):
        start = i * imgs_per_process
        end = start + imgs_per_process if i < num_processes - 1 else len(imgs_list)
        p = multiprocessing.Process(
            target=process_images,
            args=(
                start,
                end,
                imgs_list,
                save_path,
                kmeans,
                format_flag,
                video_name,
                for_kmeans_frame_no,
                progress_queue,
            ),
        )
        processes.append(p)
        p.start()

    # プロセスの処理完了を監視し、進捗を更新する
    for _ in tqdm(range(num_processes), desc="Processing images"):
        progress_queue.get()

    for p in processes:
        p.join()

    print("\033[32mクラスタごとにファイル作成完了\033[0m")


#########################################################################


# 結果をクラスタごとにディレクトリに保存
def make_cluster_dir(
    imgs_list, save_path, kmeans, format_flag, video_name, for_kmeans_frame_no
):
    # 保存先のディレクトリを空にして作成
    shutil.rmtree(save_path)
    os.mkdir(save_path)
    # クラスタごとのディレクトリ作成
    for i in tqdm(range(kmeans.n_clusters)):
        cluster_dir = save_path + "cluster{}".format(i)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

    for label, img, j in tqdm(zip(kmeans.labels_, imgs_list, for_kmeans_frame_no)):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file_number = str(j).zfill(6)
        if format_flag == "jpg":
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(label, f"{video_name}-{file_number}.jpg"),
                img,
            )
        else:
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(label, f"{video_name}-{file_number}.png"),
                img,
            )

    print("\033[32mクラスタごとにファイル作成完了\033[0m")


def create_npy_image_list(for_kmeans_array, kmeans_cnn):
# def create_npy_image_list(for_kmeans_array):
    npy_image_list = []
    # print("\033[32m配列変換中・・・\033[0m")
    print("\033[32mMobilenetで特徴量抽出中・・・\033[0m")
    for img_npy in tqdm(for_kmeans_array):
        # 画像データを一枚ずつ読み込む
        ##################################################
        # ここにMobilenetかなにかで１次元の特徴量をだすとよいかも？#
        img_npy = cv2.resize(img_npy, (224, 224))
        img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
        img_npy = img_npy[tf.newaxis]
        pred = kmeans_cnn(img_npy)
        pred = pred.numpy()
        pred = pred.flatten()
        # print(pred)
        npy_image_list.append(pred)
        ##################################################
        # normal
        # img_npy = cv2.resize(img_npy, (112, 112))
        # img_npy = img_npy.flatten()  # 一次元化
        # npy_image_list.append(img_npy / 255)  # 0~1に正規化
    # print("\033[32m配列変換完了\033[0m")
    print("\033[32m抽出完了\033[0m")
    return npy_image_list


def kmeans_main(
    save_path,
    video_name,
    for_kmeans_array,
    for_kmeans_fullframe,
    cluster_num,
    format_flag,
    for_kmeans_frame_no,
    kmeans_cnn
):
    VIDEO_NAME = video_name
    print(f"\033[32m{VIDEO_NAME}を処理しています。=>k-means\033[0m")
    # 画像データをクラスタリングした結果の保存先
    SAVE_PATH = f"{save_path}/k-means_temp/"
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    else:
        try:
            os.makedirs(SAVE_PATH)
        except OSError as e:
            print(f"Failed to create directory {SAVE_PATH}. Reason: {e}")
    print(f"\033[32m一時作業フォルダーを作成しました。\033[0m=>{SAVE_PATH}")

    # 画像データを主成分分析した結果の保存先
    CSV_PATH = f"{save_path}/{VIDEO_NAME}_pca.csv"
    # クラスターフォルダーからそれぞれランダムに抽出した保存先
    SELECTED_DIR = f"{save_path}/selected_imgs/"
    image_list = for_kmeans_fullframe
    try:
        # すでに画像データを主成分分析した結果のCSVファイルがあれば読み込む、なければexceptへ
        pca_df = pd.read_csv(CSV_PATH)
        print(f"\033[32m主成分分析ファイルを読み込みました。\033[0m=>{CSV_PATH}")
    except FileNotFoundError:
        # 画像読み込み
        print("\033[32m顔データの主成分分析を開始します。\033[0m")
        npy_image_list = create_npy_image_list(
            for_kmeans_array,
            kmeans_cnn
            )
        flag_01 = len(npy_image_list)
        if flag_01 == 0:
            messe = print(
                "\033[31m\033[1mエラー！\033[0m主成分分析が実行できませんでした。このモデルでは検知できない動画かもしれません。"
            )
            return messe

        df = pd.DataFrame(npy_image_list)
        print(df.shape)
        # 主成分分析の実行
        print("\033[32m主成分分析を実行中。\033[0m")
        pca = build_pca(df)
        pca_df = pd.DataFrame(
            pca.transform(df), columns=["PC{}".format(x + 1) for x in range(len(df))]
        )
        # plot_contribution_rate(pca)  # 累積寄与率可視化
        pca_df.to_csv(CSV_PATH, index=False)  # 保存

        print(f"\033[32m主成分分析を完了しました。\033[0m=>{CSV_PATH}")
    # kmeansによるクラスタリング
    print("\033[32mk-meansの学習を開始します。\033[0m")
    train_df = pca_df.iloc[:, :1800]  # 学習データ
    # クラスタ数を入力

    print("\033[32mモデル構築中・・・\033[0m")
    # kmeansモデル構築
    try:
        kmeans = build_kmeans(train_df, cluster_num)
    except ValueError as e:
        print(e)
        messe = print(
            "\033[31m\033[1mエラー！\033[0m抽出枚数を少なくして下さい。n_sample数以下に指定して下さい。"
        )
        return messe

    print("\033[32mモデル構築完了\033[0m")
    print("\033[32mクラスタリング中・・・\033[0m")
    # クラスタリング結果からディレクトリ作成

    # image_listをフルフレーム画像に差し替える
    make_cluster_dir_parallel(
        image_list, SAVE_PATH, kmeans, format_flag, video_name, for_kmeans_frame_no
    )
    # make_cluster_dir(image_list, SAVE_PATH, kmeans, format_flag, video_name)
    print("\033[32mクラスタリング完了\033[0m")
    pca_df["label"] = kmeans.labels_
    # クラスターフォルダーからランダムに抽出
    if not os.path.exists(SELECTED_DIR):
        try:
            os.makedirs(SELECTED_DIR)
        except OSError as e:
            print(f"Failed to create directory {SELECTED_DIR}. Reason: {e}")
    path_list = []
    for cluster_dir_path in tqdm(glob.glob(f"{SAVE_PATH}/*")):
        path_list.append(cluster_dir_path)
    path_list = sorted(path_list, key=lambda s: int(re.search(r"\d+", s).group()))
    # print(path_list)
    print("\033[32mファイル出力完了\033[0m")
    for i, img_dir in enumerate(path_list):
        img_list = glob.glob(f"{img_dir}/**.{format_flag}")
        selected_imgs = random.sample(img_list, 1)
        for img_file in selected_imgs:
            shutil.copy(img_file, SELECTED_DIR)
    # try:
    #     shutil.rmtree(SAVE_PATH)
    #     os.remove(CSV_PATH)
    # except Exception as e:
    #     print(f"Failed to delete {SAVE_PATH}. Reason: {e}")
    # print("\033[32m一時ファイルを削除しました\033[0m")
    # os.system(f"open {SELECTED_DIR}")
