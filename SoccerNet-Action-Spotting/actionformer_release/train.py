# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import json
# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, act_2_soccer, results_to_dict, post_process)
from SoccerNet.Evaluation.ActionSpotting import evaluate


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    # args.config = '/home/csgrad/mbhosale/SoccerNet-Action-Spotting/actionformer_release/configs/soccernet_i3d.yaml'
    args.print_freq = 1 # TODO Remove this later after finding correct place to update this if required.
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    
    if args.evaluate:
        val_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
        )
        # set bs = 1, and disable shuffle
        val_loader = make_data_loader(
            val_dataset, False, None, 1, cfg['loader']['num_workers'] # batch size of the model 
        )
        
        # set up evaluator
        det_eval, output_file = None, None
        if not args.saveonly:
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds']
            )
        else:
            output_file = os.path.join(ckpt_folder, 'eval_results.pkl')

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()
    
    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    eval_result_dict = {}
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            ckpt_file = 'epoch_{:03d}.pth.tar'.format(epoch + 1)
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name=ckpt_file
            )
            
            # also evluate the model
            if args.evaluate:
                print("Evaluating the model \n=> loading checkpoint '{}'".format(ckpt_file))
                
                ckpt_file = os.path.join(ckpt_folder, ckpt_file)
                checkpoint = torch.load(
                    ckpt_file,
                    map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
                )
                ckpt_model = make_meta_arch(cfg['model_name'], **cfg['model'])
                ckpt_model = nn.DataParallel(ckpt_model, device_ids=cfg['devices'])
                
                # load ema model instead
                print("Loading from EMA model ...")
                ckpt_model.load_state_dict(checkpoint['state_dict_ema'])
                del checkpoint
                
                print("\nStart testing model {:s} ...".format(cfg['model_name']))
                start = time.time()
                mAP, results = valid_one_epoch(
                    val_loader,
                    ckpt_model,
                    evaluator=det_eval,
                    output_file=output_file,
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    curr_epoch=epoch,
                    print_freq=args.print_freq
                )
                print("Actionformer evaluation done!")
                results = results_to_dict(results)
                if args.post_process and args.method == "first":
                    results = post_process(results, method="first")
                results = act_2_soccer(results)
                if args.post_process and args.method == "second":
                    results = post_process(results, method="second")
                    
                evaluation_results = evaluate(SoccerNet_path="/home/csgrad/akumar58/soccernet/spotting_data/spotting_video",
                     Predictions_path=None,
                     total_predictions=results,
                     split='test',
                     prediction_file="results_spotting.json",
                     version=2,
                     metric='tight')
                print(evaluation_results)

                eval_result_dict[epoch] = evaluation_results

                if tb_writer is not None:
                    tb_writer.add_scalar('validation/Soccernet_mAP', evaluation_results['a_mAP'], epoch)
                    tb_writer.add_scalar('validation/visible_mAP', evaluation_results['a_mAP_visible'], epoch)
                    tb_writer.add_scalar('validation/unvisible_mAP', evaluation_results['a_mAP_unshown'], epoch)
                    tb_writer.add_scalar('validation/Soccernet_precision', evaluation_results['a_precision'], epoch)
                    tb_writer.add_scalar('validation/Soccernet_recall', evaluation_results['a_recall'], epoch)
                    for i in range(len(evaluation_results['a_mAP_per_class'])):
                        tb_writer.add_scalar(f'validation/class{i}', evaluation_results['a_mAP_per_class'][i], epoch)
                print("Soccernet evaluation done!")
                end = time.time()
                print("All Evaluation done! Total time: {:0.2f} sec".format(end - start))
    
    fp = open(os.path.join(ckpt_folder, 'eval_results.txt'),'w') 
    json.dump(eval_result_dict, fp, indent=4)
    fp.close()

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate while training')
    parser.add_argument('--saveonly', action='store_true', 
                        help='')
    parser.add_argument('--post_process', action='store_true',
                        help='post process the results')
    parser.add_argument('--method', default='first',
                        help='post processing method')
    args = parser.parse_args()
    main(args)
