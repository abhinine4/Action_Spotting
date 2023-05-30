from models.i3d.extract_i3d import ExtractI3D
from utils import build_cfg_path
from omegaconf import OmegaConf
import torch
from pathlib import Path
import numpy as np
import os
from alive_progress import alive_bar

input_dir = "/home/csgrad/akumar58/soccernet/spotting_data/spotting_video/"
# input_dir = "/home/csgrad/mbhosale/SoccerNet-Action-Spotting/sn-spotting/sample_data/videos"
# output_dir = "/home/csgrad/mbhosale/SoccerNet-Action-Spotting/sn-spotting/sample_data/"
output_dir = None

if __name__ == '__main__':
    # Select the feature type
    feature_type = 'i3d'

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    torch.cuda.get_device_name(0)

    # Load and patch the config
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.video_paths = []
    for json_file in Path(input_dir).rglob('*.mkv'):
        args.video_paths.append(str(json_file))
        
    # args.show_pred = True
    # args.stack_size = 24
    # args.step_size = 24
    args.extraction_fps = 25
    args.flow_type = 'raft' # 'pwc' is not supported on Google Colab (cupy version mismatch)
    args.streams = 'flow'
    print(args)

    # Load the model
    extractor = ExtractI3D(args)

    # Extract features
    feature_dict = {}
    total = len(args.video_paths)#100
    num_videos = min(total, len(args.video_paths))
    cnt = 0
    with alive_bar(num_videos) as bar:
        for video_path in args.video_paths:
            print(f'Extracting I3D features of {video_path}')
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                savename = os.path.join(output_dir, os.path.basename(video_path)+'.npy')
            else:
                savename = os.path.join(Path(video_path).parent.absolute(), os.path.splitext(video_path)[0]+'.npy')
    
            try:
                if os.path.exists(savename):
                    data = np.load(savename)
                    print(f"I3D features already extracted for {savename}, skipping")
                    continue
            except Exception as e:
                # If an exception occurs, print the error message and skip
                print(f"An error occurred while reading the npy file: {e}, will reextract the I3D features")
                        # extract i3d features per video
            feature_dict = extractor.extract(video_path)                
            np.save(savename, feature_dict[args.streams])
            bar()
            cnt+=1
            if cnt >= num_videos:
                break
