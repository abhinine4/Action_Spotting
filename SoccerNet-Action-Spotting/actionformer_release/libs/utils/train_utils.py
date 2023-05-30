import os
import shutil
import time
import pickle
import json

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm


label2idx = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}

idx2label = {v: k for k, v in label2idx.items()}

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            # lr = scheduler.get_last_lr()[0]
            lr = 1e-4
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    # lr = scheduler.get_last_lr()[0]
    lr = 1e-4
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    losses_tracker = {}
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            outputs = model(video_list)
            output = outputs['result']
            losses = outputs['losses']
            
            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    if tb_writer is not None:
        # all losses
        tag_dict = {}
        for key, value in losses_tracker.items():
            if key != "final_loss":
                tag_dict[key] = value.val
        tb_writer.add_scalars(
            'valid/all_losses',
            tag_dict,
            curr_epoch
        )
        # final loss
        tb_writer.add_scalar(
            'valid/final_loss',
            losses_tracker['final_loss'].val,
            curr_epoch
        )


    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, _ = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/Actionformer_mAP', mAP, curr_epoch)

    return mAP, results

def load_results_from_pkl(filename):
    # load from pickle file
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def results_to_dict(results):
    """convert result arrays into dict used by json files"""
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = []

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx].append(
            {
                "label" : int(label),
                "score" : float(score),
                "segment": [float(start), float(end)],
            }
        )
    return results_dict

def act_2_soccer(res, center=True, write=False):
    # Convert the annotations from activitynet format to soccernet format
    # # Assume that the redictions are in seconds, for now.
    processed = []
    label_dir = "/home/csgrad/akumar58/soccernet/spotting_data/spotting_video/"
    final_annot = {}
    for f, segments in res.items():
        annotations = []
        nm_parts = f.split('AA')
        nm = nm_parts[-2]
        nm = " ".join(nm.split("_"))
        nm = "/".join(nm_parts[:-2]+[nm])
        total_annot = {'UrlLocal':nm, 'UrlYoutube': ''}
        if nm in processed:
            continue
        processed.append(nm)
        nm = os.path.join(label_dir, nm)
        assert os.path.exists(nm)
        for segment in segments:
            conf = segment['score']
            if '1_224p' in f or "1_I3D" in f or '1_baidu_soccer_embeddings' in f:
                game_string = "1 - "
                half = "1"
                if '1_224p' in  f:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA2_224p"
                elif '1_baidu_soccer_embeddings' in f:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA2_baidu_soccer_embeddings"
                else:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA2_I3D"
            elif '2_224p' in f or "2_I3D" in f or '2_baidu_soccer_embeddings' in f:
                game_string = "2 - "
                half = "2"
                if '2_224p' in  f:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA1_224p"
                elif '2_baidu_soccer_embeddings' in f:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA1_baidu_soccer_embeddings"
                else:
                    tmp = "AA".join(f.split("AA")[:-1])+"AA1_I3D"
            else:
                assert False
            other = None
            if tmp + "_test" in res:
                other = tmp + "_test"
                assert not tmp + "_train" in res
            elif tmp + "_train" in res:
                other = tmp + "_train"
                assert not tmp + "_test" in res
            elif tmp + "_valid" in res:
                other = tmp + "_valid"
                assert not tmp + "_valid" in res
            else:
                print(f"Other half of {f} is missing, continue")
                break # one part is missing, so continue
            assert other in res
            time_start = int(segment['segment'][0])
            time_end = int(segment['segment'][1])
            if center:
                annot = {}
                t = (time_start+time_end)/2
                mm, ss = divmod(t, 60)
                annot['gameTime'] = game_string + f"{mm:02f}:{ss:02f}"
                annot['position'] = t*1000
                annot['label'] = idx2label[segment['label']]
                annot['team'] = 'home' # hardcoded for simplicity for now.
                annot['visibility'] ='visible' # this decision likely should be taken based off the scores.
                annot['confidence'] = conf
                annot['half'] = half
                annotations.append(annot)
            else:
                for t in range(time_start, time_end+1):
                    annot = {}
                    mm, ss = divmod(t, 60)
                    annot['gameTime'] = game_string + f"{mm:02d}:{ss:02d}"
                    annot['position'] = t*1000
                    annot['label'] = idx2label[segment['label']]
                    annot['team'] = 'home' # hardcoded for simplicity for now.
                    annot['visibility'] ='visible' # this decision likely should be taken based off the scores.
                    annot['confidence'] = conf
                    annot['half'] = half
                    annotations.append(annot)
        if not other:
            continue
        segments = res[other]
        for segment in segments:
            conf = segment['score']
            if '1_224p' in other or '1_I3D' in other or '1_baidu_soccer_embeddings' in other:
                game_string = "1 - "
                half = "1"
            elif '2_224p' in other or '2_I3D' in other or '2_baidu_soccer_embeddings' in other:
                game_string = "2 - "
                half = "2"
            else:
                assert False
            time_start = int(segment['segment'][0])
            time_end = int(segment['segment'][1])
            if center:
                annot = {}
                t = (time_start+time_end)/2
                mm, ss = divmod(t, 60)
                annot['gameTime'] = game_string + f"{mm:02f}:{ss:02f}"
                annot['position'] = t*1000
                annot['label'] = idx2label[segment['label']]
                annot['team'] = 'home' # hardcoded for simplicity for now.
                annot['visibility'] ='visible' # this decision likely should be taken based off the scores.
                annot['confidence'] = conf
                annot['half'] = half
                annotations.append(annot)
            else:
                for t in range(time_start, time_end+1):
                    annot = {}
                    mm, ss = divmod(t, 60)
                    annot['gametime'] = game_string + f"{mm:02d}:{ss:02d}"
                    annot['position'] = t*1000
                    annot['label'] = idx2label[segment['label']]
                    annot['team'] = 'home' # hardcoded for simplicity for now.
                    annot['visibility'] ='visible' # this decision likely should be taken based off the scores.
                    annot['confidence'] = conf
                    annot['half'] = half
                    annotations.append(annot)
        total_annot['predictions'] = annotations
        final_annot[nm] = total_annot
        if write:
            output_file = os.path.join(nm, "results_spotting.json")
            with open(output_file, 'w') as fid:
                json.dump(total_annot, fid, indent=4)
    return final_annot

def post_process(res, method, delta=5):
    if method == 'first':
        for vdo in res.keys():
            annotations = res[vdo]
            sorted_annotations = sorted(annotations, key=lambda x: x["segment"][0])
            final_annotations = []
            i = 0
            while i < len(sorted_annotations):
                annot = sorted_annotations[i]
                t_start = annot['segment'][0]
                t_end = annot['segment'][1]
                t_label = annot['label']
                mx_conf = annot['score']
                j = i+1
                while j < len(sorted_annotations) and sorted_annotations[j]['segment'][0] <= t_start+delta:
                    if sorted_annotations[j]['label'] == t_label:
                        t_end = sorted_annotations[j]['segment'][1]
                        mx_conf = max(mx_conf, sorted_annotations[j]['score'])
                    j+=1
                i = j
                annot['segment'][1] = t_end
                annot['score'] = mx_conf
                final_annotations.append(annot)
            res[vdo] = final_annotations
    elif method == 'second':
        for vdo in res.keys():
            annotations = res[vdo]['predictions']
            sorted_annotations = sorted(annotations, key=lambda x: x["position"] + (45*60*1000)*(int(x['half'])-1)) # sort with the postions by taking into accouint the game halfs.
            final_annotations = []
            i = 0
            delta = 5
            while i < len(sorted_annotations):
                annot = sorted_annotations[i]
                t_start = t_end = annot['position']
                t_label = annot['label']
                mx_conf = annot['confidence']
                j = i+1
                while j < len(sorted_annotations) and sorted_annotations[j]['position'] <= t_start+(delta*1000):
                    if sorted_annotations[j]['label'] == t_label:
                        t_end = sorted_annotations[j]['position']
                        mx_conf = max(mx_conf, sorted_annotations[j]['confidence'])
                    j+=1
                i = j
                annot['position']= (t_start+t_end)/2
                annot['confidence'] = mx_conf
                mm, ss = divmod(annot['position']/1000, 60)
                annot['gameTime'] = annot['half'] + " - " + f"{mm:02f}:{ss:02f}"
                final_annotations.append(annot)
            res[vdo]['predictions'] = final_annotations
    return res