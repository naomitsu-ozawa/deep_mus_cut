import json
import subprocess
import platform


def webcam_list():
    os_name = platform.system()
    match os_name:
        case "Darwin":
            output = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True,
            )
            c_json = json.loads(output.stdout)

            for i, c_name in enumerate(c_json.get("SPCameraDataType", [])):
                print(f"デバイスID[{i}]:デバイス名[{c_name.get('spcamera_model-id')}]")
        case _:
            print("このオプションは、Macのみの対応です")
