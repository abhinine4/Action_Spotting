import ffmpeg
import argparse
import json
from pathlib import Path
import uuid
import os

def convert_to_srt_timestamp(ms):
    """Converts time in milliseconds to SRT timestamp format"""
    # Calculate total seconds
    total_seconds = ms // 1000
    # Calculate milliseconds
    milliseconds = ms % 1000
    # Calculate minutes and seconds
    minutes, seconds = divmod(total_seconds, 60)
    # Calculate hours and minutes
    hours, minutes = divmod(minutes, 60)
    # Return formatted timestamp string
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_to_subs(labels, subs_file):
    with open(labels, 'rb') as f:
        data = json.load(f)
    
    subtitles = ""

    for index, annotation in enumerate(data['annotations']):
        # Only 1st half (for now)
        if annotation['gameTime'].startswith("1 -"):
            start_time = int(annotation['position'])
            end_time = start_time + 1000

            start_time = convert_to_srt_timestamp(start_time)
            end_time = convert_to_srt_timestamp(end_time)

            game_time = annotation['gameTime'].split(" - ")[1]
            subtitle_text = f"{game_time} - {annotation['label']}"
            subtitle = f"{index+1}\n{start_time} --> {end_time}\n{subtitle_text}\n\n"
            subtitles += subtitle

    with open(subs_file, 'w') as f:
        f.write(subtitles)

    return

def overlay(labels, video, save_path=None):

    assert Path(labels).exists()
    # assert Path(video).exists()

    save_path = save_path if save_path else video.replace(Path(video).stem, Path(video).stem+"_overlay.mp4")
    assert save_path.endswith(".mp4")

    # Create a video stream object for the input file
    video_stream = ffmpeg.input(video)

    # generate a unique name for the file
    subs_file = str(uuid.uuid4())+".srt"

    # write subtitles to the temporary file
    convert_to_subs(labels, subs_file)

    # Add subtitles to the video stream
    # subtitles = ffmpeg.input(subtitles_file)

    # Add subtitles to the video stream
    subtitles = video_stream.filter(
        'subtitles',
        subs_file,
        force_style='Fontsize=20, PrimaryColour=&HFFFFFF&, Outline=1,Alignment=2,MarginV=5',
        # y='max(h-100-text_h-50,0)'
    )

    # subtitles = subtitles.filter('setpts', 'PTS-STARTPTS', 'TB')
    video_stream = video_stream.overlay(subtitles)

    # Add a black border at the bottom of every frame
    # video_stream = video_stream.filter('pad', w='iw', h='ih+100', x=0, y='ih')

    # Set output format and codec
    video_stream = ffmpeg.output(video_stream, save_path, vcodec='libx264', acodec='copy')

    # Run the conversion
    ffmpeg.run(video_stream)

    os.remove(subs_file)

    return


def main():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process command line arguments')

    # add arguments
    parser.add_argument('--labels', type=str, required=True, help='path to labels json file')
    parser.add_argument('--video', type=str, required=True, help='path to input video file')
    parser.add_argument('--save_path', type=str, help='path to save annotated video file, defaults to video+"_overlay.mp4"')
    
    # parse command line arguments
    args = parser.parse_args()

    overlay(args.labels, args.video, args.save_path)

    
# python overlay_annotation.py --labels /path/to/labels.json --video /path/to/video.mkv --save_path /path/to/save.mp4


if __name__ == '__main__':
    main()
