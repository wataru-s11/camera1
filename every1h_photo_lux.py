from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable, Optional

import requests


DEFAULT_SHARE_ROOT = PureWindowsPath(r"Z:\\Raspi_face")
CAPTURE_DATE_ENV = "CAMERA_CAPTURE_DATE"
SAVE_ROOT_PREFIX_ENV = "CAMERA_SAVE_ROOT_PREFIX"
CYCLE_INTERVAL_ENV = "CAPTURE_CYCLE_SECONDS"


def _env_interval(name: str, default: int) -> int:
    """Read a positive integer interval from the environment."""

    value = os.environ.get(name)
    if not value:
        return default

    try:
        seconds = int(value)
    except ValueError:
        print(f"[WARN] Invalid {name} value: {value!r}. Using default {default} seconds.")
        return default

    if seconds <= 0:
        print(f"[WARN] {name} must be positive. Using default {default} seconds.")
        return default

    return seconds


CYCLE_INTERVAL_SECONDS = _env_interval(CYCLE_INTERVAL_ENV, 1800)


def _z_drive_path(*parts: str) -> str:
    return str(DEFAULT_SHARE_ROOT.joinpath(*parts))


def resolve_capture_date(reference: Optional[datetime] = None) -> str:
    """Determine the folder name used for captures on a given day."""

    override = os.environ.get(CAPTURE_DATE_ENV)
    if override:
        return override

    reference = reference or datetime.now()
    return reference.strftime("%Y%m%d")


def _apply_save_root_override(save_root: str) -> str:
    """Allow redirecting network share paths via ``SAVE_ROOT_PREFIX_ENV``."""

    override_prefix = os.environ.get(SAVE_ROOT_PREFIX_ENV)
    if not override_prefix:
        return save_root

    pure_save_root = PureWindowsPath(save_root)
    try:
        relative = pure_save_root.relative_to(DEFAULT_SHARE_ROOT)
    except ValueError:
        return str(Path(override_prefix) / pure_save_root.name)

    combined = Path(override_prefix)
    for part in relative.parts:
        combined /= part
    return str(combined)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Connection and storage details for a single camera."""

    name: str
    pi_hostname: str
    save_root: str
    esp32_ip: str
    date_subfolder: bool = True

    def label(self) -> str:
        return self.name or self.pi_hostname

    def resolve_save_folder(self, cycle_time: Optional[datetime] = None) -> Path:
        if cycle_time is None:
            cycle_time = datetime.now()

        root = Path(_apply_save_root_override(self.save_root))
        if not self.date_subfolder:
            return root

        capture_date = resolve_capture_date(cycle_time)
        return root / capture_date


CAMERA_CONFIGS: tuple[CameraConfig, ...] = (
    CameraConfig(
        name="pi-vital2",
        pi_hostname="pi-vital2.local",
        save_root=_z_drive_path("pi2"),
        esp32_ip="http://192.168.1.213",
    ),
    CameraConfig(
        name="pi-vital3",
        pi_hostname="pi-vital3.local",
        save_root=_z_drive_path("pi3"),
        esp32_ip="http://192.168.1.214",
    ),
    CameraConfig(
        name="pi-vital4",
        pi_hostname="pi-vital4.local",
        save_root=_z_drive_path("pi4"),
        esp32_ip="http://192.168.1.215",
    ),
    CameraConfig(
        name="pi-vital5",
        pi_hostname="pi-vital5.local",
        save_root=_z_drive_path("pi5"),
        esp32_ip="http://192.168.1.216",
    ),
)


@dataclass(slots=True)
class CaptureResult:
    """Details about a capture cycle for a single camera."""

    image_path: str
    lux_path: str
    timestamp: str


def capture_and_transfer(
    target: CameraConfig | os.PathLike[str] | str,
    esp32_endpoint: Optional[str] = None,
    pi_hostname: Optional[str] = None,
    *,
    camera_name: Optional[str] = None,
    cycle_time: Optional[datetime] = None,
) -> CaptureResult:
    """Capture an image and lux reading, then transfer the assets to storage."""

    if isinstance(target, CameraConfig):
        config = target
        label = camera_name or config.label()
    else:
        if esp32_endpoint is None or pi_hostname is None:
            raise ValueError(
                "esp32_endpoint and pi_hostname are required when passing a raw path."
            )
        config = CameraConfig(
            name=camera_name or (pi_hostname or ""),
            pi_hostname=pi_hostname,
            save_root=str(Path(target)),
            esp32_ip=esp32_endpoint,
            date_subfolder=False,
        )
        label = config.label()

    cycle_time = cycle_time or datetime.now()

    save_folder = config.resolve_save_folder(cycle_time)
    save_folder.mkdir(parents=True, exist_ok=True)

    timestamp = cycle_time.strftime("%Y%m%d_%H%M")
    image_filename = f"image_{timestamp}.jpg"
    lux_filename = f"lux_{timestamp}.txt"

    lux_path = save_folder / lux_filename
    image_path = save_folder / image_filename

    try:
        response = requests.get(config.esp32_ip, timeout=5)
        response.raise_for_status()
        lux_text = response.text.strip()
    except Exception as exc:
        lux_text = f"Error: {exc}"
        print(f"[WARN] {label}: Failed to fetch lux value ({exc})")

    with open(lux_path, "w", encoding="utf-8") as lux_file:
        lux_file.write(lux_text)

    ssh_command = (
        f'ssh sakai@{config.pi_hostname} "libcamera-jpeg -o /home/sakai/{image_filename}"'
    )
    scp_command = (
        f'scp sakai@{config.pi_hostname}:/home/sakai/{image_filename} "{image_path}"'
    )

    try:
        subprocess.run(ssh_command, shell=True, check=True)
        subprocess.run(scp_command, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label}: Command failed (exit code {exc.returncode}): {exc.cmd}"
        ) from exc

    print(f"[INFO] {label}: Capture complete ({timestamp})")
    return CaptureResult(
        image_path=str(image_path), lux_path=str(lux_path), timestamp=timestamp
    )


def main() -> None:
    while True:
        cycle_started = datetime.now()
        print(f"[INFO] Capture cycle started at {cycle_started:%Y-%m-%d %H:%M:%S}")

        ensure_save_directories(CAMERA_CONFIGS, cycle_started)

        for config in CAMERA_CONFIGS:
            label = config.label()
            try:
                capture_and_transfer(config, cycle_time=cycle_started)
            except Exception as exc:
                print(f"[ERROR] {label}: {exc}")

        print(f"[INFO] Cycle finished. Sleeping for {CYCLE_INTERVAL_SECONDS} seconds.")
        time.sleep(CYCLE_INTERVAL_SECONDS)


def ensure_save_directories(
    configs: Iterable[CameraConfig], reference_time: Optional[datetime] = None
) -> None:
    """Pre-create the save directories for the provided configurations."""

    reference_time = reference_time or datetime.now()
    for config in configs:
        folder = config.resolve_save_folder(reference_time)
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
