import subprocess
from datetime import datetime
import requests
import os
import time

# 保存フォルダ
save_folder = "C:/Users/sakai/OneDrive/Desktop/Raspi5/pi-vital5/20250831" 
os.makedirs(save_folder, exist_ok=True)
# ESP32 IP
esp32_ip = "http://192.168.1.4"

while True:
    # 1. 日時付きファイル名を作成
    now = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"image_{now}.jpg"
    lux_filename = f"lux_{now}.txt"

    # 2. 照度取得
    try:
        lux = requests.get(esp32_ip, timeout=5).text
    except Exception as e:
        lux = f"Error: {e}"

    # 3. 照度を保存
    with open(os.path.join(save_folder, lux_filename), "w") as f:
        f.write(lux)

    # 4. 撮影
    ssh_command = f'ssh sakai@pi-vital5.local "libcamera-jpeg -o /home/sakai/{filename}"'
    subprocess.run(ssh_command, shell=True)

    # 5. 画像をPCへ転送
    scp_command = f'scp sakai@pi-vital5.local:/home/sakai/{filename} "{os.path.join(save_folder, filename)}"'
    subprocess.run(scp_command, shell=True)

    print(f"[{now}] 撮影・照度保存完了")

    # 6. 1時間待機 (3600秒)
    time.sleep(3600)
