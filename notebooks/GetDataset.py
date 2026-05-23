"""Friendly webcam dataset collector for ASL hand images.

Run interactively:
    python notebooks/GetDataset.py

Or skip prompts:
    python notebooks/GetDataset.py --label A --name chris --max-images 200
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys
import time

import cv2


os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asl_to_text.hands import HandExtractor  # noqa: E402


@contextmanager
def suppress_native_stderr():
    """Hide noisy native library startup logs while MediaPipe initializes."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def clean_filename_part(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return value.strip("_")


def ask_if_missing(value: str | None, prompt: str, *, default: str | None = None) -> str:
    if value:
        return value

    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    if not answer and default is not None:
        return default
    return answer


def draw_status(frame, label: str, saved: int, output_dir: Path, paused: bool, detecting: bool) -> None:
    height = frame.shape[0]
    status = "PAUSED" if paused else ("HAND DETECTED" if detecting else "NO HAND")
    color = (0, 190, 255) if paused else ((55, 220, 120) if detecting else (70, 70, 255))

    lines = [
        f"Label: {label.upper()}   Saved: {saved}",
        f"Status: {status}",
        "q: quit   p: pause/resume",
        f"Folder: {output_dir}",
    ]

    y = 30
    for index, line in enumerate(lines):
        line_color = color if index == 1 else (245, 245, 245)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)
        y += 28

    cv2.putText(
        frame,
        "Keep your full hand inside the green box.",
        (14, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        2,
    )


def collect_images(
    label: str,
    name: str,
    output_root: Path,
    camera: int,
    margin: int,
    max_images: int | None,
    save_interval: float,
    countdown: int,
) -> int:
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera}. Try another camera with --camera 1."
        )

    with suppress_native_stderr():
        extractor = HandExtractor(
            margin=margin,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    saved = 0
    paused = False
    last_save_time = 0.0
    window_name = "ASL Dataset Collector"

    print()
    print(f"Saving cropped hand images to: {output_dir}")
    print("Controls: press 'q' to quit, 'p' to pause/resume.")
    if max_images:
        print(f"Collection will stop after {max_images} saved images.")
    if countdown > 0:
        print(f"Starting in {countdown} seconds...")
        time.sleep(countdown)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera opened, but no frame could be captured.")

            frame = cv2.flip(frame, 1)
            crop = extractor.extract(frame, draw=True)
            now = time.monotonic()
            can_save = now - last_save_time >= save_interval

            if crop is not None and not paused and can_save:
                filename = output_dir / f"{label}_{saved:04d}_{name}.jpg"
                if cv2.imwrite(str(filename), crop.image):
                    saved += 1
                    last_save_time = now
                    print(f"Saved {saved:04d}: {filename}")
                else:
                    print(f"Warning: failed to write {filename}")

            draw_status(frame, label, saved, output_dir, paused, crop is not None)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused

            if max_images is not None and saved >= max_images:
                print(f"Reached --max-images limit ({max_images}).")
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture cropped webcam hand images for an ASL training label."
    )
    parser.add_argument(
        "--label",
        help="Label/class to collect, for example A, B, space, or nothing.",
    )
    parser.add_argument("--name", help="Name or initials to include in each filename.")
    parser.add_argument(
        "--output",
        default="dataset/custom",
        help="Dataset root folder. Images are saved under <output>/<label>.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use.")
    parser.add_argument(
        "--margin",
        type=int,
        default=100,
        help="Pixels of padding around the hand crop.",
    )
    parser.add_argument("--max-images", type=int, help="Stop after saving this many images.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.15,
        help="Minimum seconds between saved frames.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to wait before recording starts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("ASL Dataset Collector")
    print("---------------------")
    label = clean_filename_part(ask_if_missing(args.label, "Letter/sign label"))
    name = clean_filename_part(ask_if_missing(args.name, "Your name or initials", default="user"))

    if not label:
        raise SystemExit("Label cannot be empty.")
    if not name:
        raise SystemExit("Name cannot be empty.")
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be greater than 0.")
    if args.interval < 0:
        raise SystemExit("--interval cannot be negative.")
    if args.countdown < 0:
        raise SystemExit("--countdown cannot be negative.")

    try:
        count = collect_images(
            label=label,
            name=name,
            output_root=Path(args.output),
            camera=args.camera,
            margin=args.margin,
            max_images=args.max_images,
            save_interval=args.interval,
            countdown=args.countdown,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print()
    print(f"Done. Saved {count} images for '{label}' in {Path(args.output) / label}.")


if __name__ == "__main__":
    main()
