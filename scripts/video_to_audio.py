import subprocess
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"


# ============================================================
# CONVERT VIDEO TO MP3
# ============================================================

def convert_video_to_audio(video_path, output_path):

    print("\n" + "=" * 70)
    print("VIDEO → AUDIO CONVERSION")
    print("=" * 70)

    print(f"\nInput video:")
    print(video_path)

    print(f"\nOutput audio:")
    print(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        "-y",
        str(output_path)
    ]

    subprocess.run(
        command,
        check=True
    )

    print("\nConversion completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    VIDEO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find video files
    # --------------------------------------------------------

    videos = list(
        VIDEO_DIR.glob("*.webm")
    )

    if not videos:

        print(
            "\nNo WEBM video found in:"
        )

        print(VIDEO_DIR)

        return

    # --------------------------------------------------------
    # Use the first source video as our demo
    # --------------------------------------------------------

    video_path = videos[0]

    output_name = (
        video_path.stem
        + ".mp3"
    )

    output_path = (
        AUDIO_DIR
        / output_name
    )

    # --------------------------------------------------------
    # Avoid unnecessary reconversion
    # --------------------------------------------------------

    if output_path.exists():

        print(
            "\nAudio file already exists:"
        )

        print(output_path)

        print(
            "\nSkipping conversion."
        )

        return

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    convert_video_to_audio(
        video_path,
        output_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()