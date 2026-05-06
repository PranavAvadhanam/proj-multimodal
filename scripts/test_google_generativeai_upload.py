from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload an MP4 and ask Gemini for audio transcription.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to local raw MP4 file.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model id.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    video_file = genai.upload_file(str(video_path))
    # google.generativeai upload is async; wait until file is usable.
    for _ in range(60):
        refreshed = genai.get_file(video_file.name)
        state_name = getattr(getattr(refreshed, "state", None), "name", "")
        if state_name == "ACTIVE":
            video_file = refreshed
            break
        if state_name == "FAILED":
            raise RuntimeError(f"Uploaded file failed processing: {refreshed.name}")
        time.sleep(1)
    else:
        raise TimeoutError("Uploaded file never reached ACTIVE state within 60s.")

    response = model.generate_content(
        [
            video_file,
            "Ignore the visuals entirely. What words are spoken in this video? Transcribe the speech.",
        ]
    )
    print(response.text)


if __name__ == "__main__":
    main()
