import os
import cv2
import sys
from pathlib import Path
from tqdm import tqdm
from subprocess import Popen, PIPE
from PIL import Image
import re

buff_path = Path("./buff")
output_path = Path("./output")


def get_bitrate(height, width, duration, bitrate, target_size):
    # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    min_audio_bitrate = height * width // 100
    max_audio_bitrate = height * width
    target_total_bitrate = (target_size * 1024 * 8) / (1.073741824 * duration)
    # Target audio bitrate, in bps
    if 10 * bitrate > target_total_bitrate:
        audio_bitrate = target_total_bitrate / 10
        if audio_bitrate < min_audio_bitrate < target_total_bitrate:
            bitrate = min_audio_bitrate
        elif audio_bitrate > max_audio_bitrate:
            bitrate = max_audio_bitrate
    video_bitrate = target_total_bitrate - bitrate
    return video_bitrate


def stream_video(output_file: Path, device=0, fps=24, codec="mp4v", **kwargs):
    output_path.mkdir(parents=True, exist_ok=True)
    buff_path.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(device)
    cam.set(cv2.CAP_PROP_FPS, fps)
    image_quality = kwargs.get("quality", 50)  # Setting default image quality to 50

    if not cam.isOpened():
        print("Unable to open camera")
        sys.exit(-1)

    width = int(cam.get(3))
    height = int(cam.get(4))

    target_size = (width * height) * (1.073741824 * 1 / fps) / (8 * 1024)
    bitrate = int(get_bitrate(height, width, 1 / fps, cam.get(cv2.CAP_PROP_BITRATE), target_size) / 1000)

    p = Popen(
        ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(fps), '-i', '-', '-vcodec', 'h264', '-qscale',
         '5', '-r', str(fps), '-b:v', str(bitrate) + "K", '-maxrate', str(bitrate) + "K", "-bufsize",
         str(int(bitrate // 2)) + "K", 'video.mp4'], stdin=PIPE)

    codec = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(str(output_file), codec, fps, (width, height))
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 10)

    image_count = 0
    while True:
        check, frame = cam.read()

        if check:
            out.write(frame)
            cv2.imshow('video', frame)

            # Write to buffer
            cv2.imwrite(
                str(buff_path) + f"/{image_count}_frame.jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, image_quality]
            )
            image_count += 1

            # Write to stdin
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.save(p.stdin, 'JPEG')

            key = cv2.waitKey(1)
            # exit on esc
            if key == 27:
                break
        else:
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


def compose_video(output_file: Path, image_folder: Path, fps=24, codec="mp4v", size=(1920, 1080)):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(str(output_file), fourcc, fps, size, isColor=True)

    images_list = sorted([file for file in os.listdir(image_folder) if file.endswith('.jpg')], key=lambda x: int(re.search(r"[0-9]+", x)[0]))
    print(images_list)
    for image_name in tqdm(images_list):
        image_full_path = os.path.join(str(image_folder), image_name)
        image = cv2.imread(image_full_path)
        image = cv2.resize(image, size)
        video.write(image)

    buff_files = image_folder.glob("*")

    for file in buff_files:
        os.remove(file)

    video.release()


if __name__ == '__main__':
    stream_video(output_path / "output.mp4", device=1)
    compose_video(output_path / "composed.mp4", buff_path)
