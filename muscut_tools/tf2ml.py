# %%
from pathlib import Path

import coremltools as ct
import tensorflow as tf

# %%
def tf2ml(input_path):
    print("\033[32mTensorFlowモデルをCoreMLモデルへ変換します。\033[0m")
    in_path = Path(input_path)
    out_path = f"{in_path.parent}/{in_path.stem}.mlmodel"
    print(out_path)
    tf_model = tf.keras.models.load_model(input_path)
    ct_model = ct.convert(tf_model)
    ct_model.save(out_path)
