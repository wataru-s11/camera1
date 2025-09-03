import os
import subprocess
from datetime import datetime
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Constants
SAVE_ROOT = r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital5"
ESP32_IP = "http://192.168.1.4"
REMOTE_HOST = "sakai@pi-vital5.local"
MODEL_PATH = r"C:\Users\sakai\OneDrive\Desktop\Raspi5\face_annotation\yolov8n.pt"
CROP_REGION = (475, 43, 2728, 2383)  # x, y, w, h


def get_lux(ip):
    try:
        return requests.get(ip, timeout=5).text
    except Exception as e:
        return f"Error: {e}"


def capture_image(remote, filename):
    ssh_cmd = f'ssh {remote} "libcamera-jpeg -o /home/sakai/{filename}"'
    subprocess.run(ssh_cmd, shell=True)


def transfer_image(remote, filename, dest):
    scp_cmd = f'scp {remote}:/home/sakai/{filename} "{dest}"'
    subprocess.run(scp_cmd, shell=True)


def apply_gamma(image, lux):
    try:
        lux_value = float(lux)
        if lux_value < 30:
            gamma = 1.5
        elif lux_value < 80:
            gamma = 1.2
        elif lux_value > 120:
            gamma = 0.9
        else:
            gamma = 1.0
    except Exception:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)


def main():
    # Timestamp and folder setup
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SAVE_ROOT, now)
    os.makedirs(session_dir, exist_ok=True)

    image_fname = f"image_{now}.jpg"
    lux_fname = f"lux_{now}.txt"
    processed_fname = f"processed_{now}.jpg"

    # 1. Measure illuminance
    lux = get_lux(ESP32_IP)
    with open(os.path.join(session_dir, lux_fname), "w") as f:
        f.write(lux)

    # 2. Capture image and transfer
    capture_image(REMOTE_HOST, image_fname)
    local_image_path = os.path.join(session_dir, image_fname)
    transfer_image(REMOTE_HOST, image_fname, local_image_path)

    # 3. Crop and brightness correction
    img = cv2.imread(local_image_path)
    if img is None:
        print("画像読み込み失敗")
        return
    x, y, w, h = CROP_REGION
    cropped = img[y:y+h, x:x+w]
    corrected = apply_gamma(cropped, lux)
    processed_path = os.path.join(session_dir, processed_fname)
    cv2.imwrite(processed_path, corrected)

    # 4. Face detection and crop
    model = YOLO(MODEL_PATH)
    pil_img = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    results = model(pil_img)

    face_idx = 0
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face = corrected[y1:y2, x1:x2]
            face_idx += 1
            face_path = os.path.join(session_dir, f"face_{now}_{face_idx}.jpg")
            cv2.imwrite(face_path, face)
            print(f"保存完了: {face_path}")

    print(f"処理完了: {session_dir}")


if __name__ == "__main__":
    main()
