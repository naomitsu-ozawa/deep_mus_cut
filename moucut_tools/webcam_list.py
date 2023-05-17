import json
import subprocess


def webcam_list():
    output = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True
    )
    c_json = json.loads(output.stdout)

    for i, c_name in enumerate(c_json.get("SPCameraDataType", [])):
        print(f"デバイスID[{i}]:デバイス名[{c_name.get('spcamera_model-id')}]")
