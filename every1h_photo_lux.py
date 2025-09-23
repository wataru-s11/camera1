import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


CYCLE_INTERVAL_SECONDS = 1800

CAMERA_CONFIGS = [
    {
        "name": "pi-vital2",
        "pi_hostname": "pi-vital2.local",
        "save_folder": r"Z:\\Raspi_face\\pi2\\20250918",
        "esp32_ip": "http://192.168.1.213",
    },
    {
        "name": "pi-vital3",
        "pi_hostname": "pi-vital3.local",
        "save_folder": r"Z:\\Raspi_face\\pi3\\20250918",
        "esp32_ip": "http://192.168.1.214",
    },
    {
        "name": "pi-vital4",
        "pi_hostname": "pi-vital4.local",
        "save_folder": r"Z:\\Raspi_face\\pi4\\20250918",
        "esp32_ip": "http://192.168.1.215",
    },
    {
        "name": "pi-vital5",
        "pi_hostname": "pi-vital5.local",
        "save_folder": r"Z:\\Raspi_face\\pi5\\20250918",
        "esp32_ip": "http://192.168.1.216",
    },
]


@dataclass(slots=True)
class CaptureResult:
    """Details about a capture cycle for a single camera."""

    image_path: str
    lux_path: str
    timestamp: str


def capture_and_transfer(
    save_folder: str,
    esp32_endpoint: str,
    pi_hostname: str,
    *,
    camera_name: Optional[str] = None,
) -> CaptureResult:
    """Capture an image and lux reading, then transfer the assets to ``save_folder``."""
    label = camera_name or pi_hostname

    os.makedirs(save_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    image_filename = f"image_{timestamp}.jpg"
    lux_filename = f"lux_{timestamp}.txt"

    lux_path = os.path.join(save_folder, lux_filename)
    image_path = os.path.join(save_folder, image_filename)

    try:
        response = requests.get(esp32_endpoint, timeout=5)
        response.raise_for_status()
        lux_text = response.text.strip()
    except Exception as exc:
        lux_text = f"Error: {exc}"
        print(f"[WARN] {label}: Failed to fetch lux value ({exc})")

    with open(lux_path, "w", encoding="utf-8") as lux_file:
        lux_file.write(lux_text)

    ssh_command = f'ssh sakai@{pi_hostname} "libcamera-jpeg -o /home/sakai/{image_filename}"'
    scp_command = f'scp sakai@{pi_hostname}:/home/sakai/{image_filename} "{image_path}"'

    try:
        subprocess.run(ssh_command, shell=True, check=True)
        subprocess.run(scp_command, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label}: Command failed (exit code {exc.returncode}): {exc.cmd}"
        ) from exc

    print(f"[INFO] {label}: Capture complete ({timestamp})")
    return CaptureResult(image_path=image_path, lux_path=lux_path, timestamp=timestamp)


def main() -> None:
    for config in CAMERA_CONFIGS:
        os.makedirs(config["save_folder"], exist_ok=True)

    while True:
        cycle_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Capture cycle started at {cycle_started}")

        for config in CAMERA_CONFIGS:
            name = config["name"]
            try:
                capture_and_transfer(
                    config["save_folder"],
                    config["esp32_ip"],
                    config["pi_hostname"],
                    camera_name=name,
                )
            except Exception as exc:
                print(f"[ERROR] {name}: {exc}")

        print(f"[INFO] Cycle finished. Sleeping for {CYCLE_INTERVAL_SECONDS} seconds.")
        time.sleep(CYCLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
