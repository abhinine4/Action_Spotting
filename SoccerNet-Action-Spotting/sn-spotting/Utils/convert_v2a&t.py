import whisper
import moviepy.editor as mp
import argparse
import os
import pathlib

# convert the video to audio and text.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--video_format', default='.mkv', type=str, help='file format of video files in dataset.')
    parser.add_argument('--audio', type=bool, default=True)
    parser.add_argument('--audio_format', default='.wav', type=str, help='file format of output audio files, if --audio is set to True')
    parser.add_argument('--text', type=bool, default=True)
    parser.add_argument('--text_format', default='.txt', type=str, help='file format of transcribed text files, if --text is set to True')    
    parser.add_argument('--out_path', default=None, help='output folder')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    
    # implement other formats if and when required
    assert args.video_format in ['.mkv']
    assert args.audio_format in ['.wav']
    assert args.text_format in ['.txt']
    loaded = False
    
    for f in list(pathlib.Path(args.dataset_path).rglob("*")):
        if pathlib.Path(f).suffix != ".mkv":
            continue
        out_path = pathlib.Path(f).parent.absolute()
        if args.out_path is not None:
            out_path = os.path.join(args.out_path, out_path)
        clip = mp.VideoFileClip(str(f))
        if args.audio:
            audio_file = pathlib.Path(f).stem + args.audio_format
            clip.audio.write_audiofile(os.path.join(out_path, audio_file))
            if args.text:        
                if not loaded:
                    model = whisper.load_model("medium")
                    loaded = True
                text_file = os.path.join(out_path, pathlib.Path(audio_file).stem + args.text_format)
                result = model.transcribe(os.path.join(out_path, audio_file))
                with open(text_file, 'wt') as t:
                    t.write(result["text"])
        # elif not args.audio and args.text:
        #     audio_file = pathlib.Path(f).stem + args.audio_format
        #     if args.text:        
        #         if not loaded:
        #             model = whisper.load_model("medium")
        #             loaded = True
        #         text_file = pathlib.Path(audio_file).stem + args.text_format
        #         result = model.transcribe(os.path.join(out_path, audio_file))
        #         with open(text_file, 'wt') as t:
        #             t.write(result["text"])