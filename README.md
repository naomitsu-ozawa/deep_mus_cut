# deep_mou_cut_2
## 動画からマウスの顔をいい感じで画像で保存するアプリ  

https://github.com/naomitsu-ozawa/deep_mou_cut_2/assets/129124821/ea1ff869-2e6b-4ba1-9224-e11d7a950d8a

## 説明
映像内のマウスの顔を検知していい感じに切り取ってくれます。  
現在、C57BL/6などの黒マウスの横顔に対応しています。  
  
### 動作フロー
```mermaid
%%{init:{'theme':'forest'}}%%
graph TD
    A[imput] -->|解析したい動画| B(物体検知\nYolov8 Custom model)
    B -->|検知された部位の画像| C(画像分類\nMobile_net V3 Custom model)
    C -->|分類された画像| D(クラスタリング\nk-means)
    D -->|クラスタリングされた画像| E("random.sample([k-means-images],1)")
    E -->|各クラスタから一枚づつ出力| F[output]
    A --> |抽出したい枚数を\nクラスタ数として渡す|　D
```
- 物体検知と画像分類モデルをCoreMlへ変換したモデルをDefaultで動作するようにしています。
- 物体検知（PyTorch）・画像分類（TensorFlow ）のモデルも同梱しているので、Windows/Linux/macでの動作が可能です。オプションで指定して下さい。
---

## インストール
- python3.10~で動作します。
- conda等で仮想環境を作成して下さい。
1. リポジトリをクローンします。  
   ```git clone https://github.com/naomitsu-ozawa/deep_mou_cut_2.git```
2. Ultralyticsをインストールします。  
   ```pip install ultralytics```
3. Scikit-learnをインストールしてください。  
```pip install scikit-learn```  
1. CoreMLに対応したMacの場合は、CoreMLtoolsをインストールします。  
   ```pip install coremltools```  
2. CoreML非対応のWindows/Linux/macで利用する場合は、Tensorflowをインストールします。  
```pip install tensorflow```  
macの場合は以下もインストールする。  
```pip install tensorflow-metal```    
  
- numpyでエラーが起こる場合は、pipの方のnumpyを更新します。  
```pip install -U numpy```  
### アップデート方法 
- deep_mou_cut_2フォルダに移動後、git pullして下さい。
---
### 使い方
  
次のコマンドで解析が始まります。  
```python moucut.py -f <movie_file_path>```  
  
変数に解析したい動画のパスを入れて指定することもできます。  
```
movie="<movie_file_path>"  
python moucut.py -f $movie
```    
  
顔検知中のプレビューを表示させるには、-sオプションをつけて下さい。  
```python moucut.py -f $movie -s```  


### 顔検出でGPUを使う方法
-m　オプションに"tf"を指定して下さい。
-d　オプションに "cuda"を指定して下さい。 
（Macの場合は"mps"）  
```python moucut.py -f ＄movie -m tf -d cuda```  
  
CNNモデルの分類ではTensorFlowを使っています。  
GPUを利用したい場合、お使いのプラットフォームに合わせたTensorFlowを環境にインストールしてください。  
  
  
---
### オプション
| option | description |  
| ---- | ---- |
| -f,--file | 解析したいファイルのパス（必須）[file_path,webcam]<br>-f <file_path>を指定すると動画ファイルの解析を行います。<br>-f webcam0を指定するとデバイスID：０のカメラに接続できます。(テスト機能)<br>複数台カメラが接続されている場合は、webcam*の番号を変更してみて下さい。 |
| -m,--mode | モード[coreml,tf]<br>-m coreml：物体検知と画像分類にCore MLを利用します。<br>-m tf：物体検知にPyTorch、画像分類にTensorFlowを利用します。 |
| -d,--device | 物体検知部分で利用するデバイス名 [cpu,cuda,mps]<br>--mode tfの時のPyTrochデバイスを指定できます。|
| -t,--tool | 使用するツール名 [default,kmeans_image_extractor,all_extract]<br>-t default：未指定と同じ動作になります。<br>-t kmeans_image_extractor：動画からk-meansアルゴリズムを利用して指定枚数のフレーム画像を抽出します。<br>-t all_extract：検知された顔を全て画像として保存します。<br><br> |
| -i,--image_format | 出力画像のフォーマット [jpg,png]<br>-i png：デフォルトです。未指定と同じ動作になります。<br>-i jpg：JPEG形式で保存します。容量を節約したい場合に有効です。 |
| -s,--show | プレビューモード |
| -n,--number | 抽出枚数 |
| -wc,--without_cnn | 画像分類を行わずに解析します。 |
  
#### modeについて
Defaultは”CoreML”で動作するようになっています。 CoreML非対応の環境で動作せる場合は、”tf”を指定して下さい。
| --mode | 詳細 |
| ---- | ---- |
| coreml | 物体検出と画像分類にCoreMlモデルを使用します。(default) |
| tf | 物体検出と画像分類にPyTrochとTensorFlowを使用します。 |

#### deviceについて
モード”tf”時の物体検出で利用するPyTorchデバイスを指定できます。
| --device | 詳細 |
| ---- | ---- |
| cpu | 物体検知にcpuを使います。(default) |
| cuda | 物体検知にCUDAを使います。（n VidiaのGPUが必要です。） |
| mps | AppleのMetal Performance Shadersを使います。 |

#### toolについて
| --tool | 詳細 |
| ---- | ---- |
| all_extract | 動画からマウスの顔を全て抽出します。|
| kmeans_image_extractor | k-meansアルゴリズムを使って動画から指定枚数の画像を抽出し|ます。
  
#### 保存できるフォーマットについて  
- オプションを指定しない場合は、png形式で保存されます。オプションで指定することでjpg形式で保存可能です。
---
### その他動物への対応
お問い合わせください。
