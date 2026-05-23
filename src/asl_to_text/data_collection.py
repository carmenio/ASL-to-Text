"""Command-line webcam data collection for ASL training images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .hands import HandExtractor


def collect_data(label: str, name: str, output: str | Path, camera: int = 0) -> int:
    output_dir = Path(output) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera}.")

    extractor = HandExtractor(static_image_mode=False)
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to capture frame from webcam.")

            frame = cv2.flip(frame, 1)
            crop = extractor.extract(frame, draw=True)
            if crop is not None:
                filename = output_dir / f"{label}_{frame_count:04d}_{name}.jpg"
                cv2.imwrite(str(filename), crop.image)
                frame_count += 1
                cv2.putText(
                    frame,
                    f"Saved {frame_count} images for {label}",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (55, 220, 120),
                    2,
                )

            cv2.imshow("ASL-to-Text Data Collection - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    return frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture cropped hand images for ASL training.")
    parser.add_argument("--label", required=True, help="Class label folder to write, for example A or space.")
    parser.add_argument("--name", required=True, help="Name or initials to include in generated filenames.")
    parser.add_argument("--output", default="data/raw", help="Dataset root directory.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = collect_data(args.label, args.name, args.output, args.camera)
    print(f"Saved {count} images to {Path(args.output) / args.label}")


if __name__ == "__main__":
    main()
