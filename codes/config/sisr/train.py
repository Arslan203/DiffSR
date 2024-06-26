import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import os.path as osp
from basicsr.metrics import calculate_metric
from basicsr.data.prefetch_dataloader import CUDAPrefetcher
from basicsr.utils import init_wandb_logger, init_tb_logger

# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset, dataloader_iterable
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt.get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt.get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def main():
    #### setup options of three networks
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="D:/EDiffSR/codes/config/sisr/options/setting.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    opt['root_path'] = root_path

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # os.system("rm ./log")
            # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=opt['logger'].get('show_logs', False),
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=opt['logger'].get('show_logs', False),
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        tb_logger = init_tb_loggers(opt)
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    error = mp.Value('b', False)

    prefetcher = None
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        
    # make iterable from val_dataloader
    val_loader_iter = dataloader_iterable(val_loader)
    eval_FID = opt.get('FID') is not None
    if eval_FID:
        FID_dataloader = []
    val_size = opt['train'].get('val_size', 2000)

    # -------------------------------------------------------------------------
    # -------------------------正式开始训练，前面都是废话---------------------------
    # -------------------------------------------------------------------------
    if prefetcher is not None:
        for epoch in range(start_epoch, total_epochs + 1):
            if opt["dist"]:
                train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()
            while train_data is not None:
                current_step += 1

                if current_step > total_iters:
                    break

                LQ, GT = train_data["LQ"], train_data["GT"]  #  b 3 32 32; b 3 128 128

                LQ = util.upscale(LQ, scale)  #  bicubic, which can be repleced by deep networks

                # random timestep and state (noisy map) via SDE
                timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)  # t=batchsize，states [b 3 128 128]

                model.feed_data(states, LQ, GT)  # xt, mu, x0, 将加了噪声的LR图xt，LR以及GT输入改进的UNet进行去噪

                model.optimize_parameters(current_step, timesteps, sde)  # 优化UNet

                model.update_learning_rate(
                    current_step, warmup_iter=opt["train"]["warmup_iter"]
                )

                if current_step % opt["logger"]["print_freq"] == 0:
                    logs = model.get_current_log_reset(sde, opt['logger']['print_freq'])
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

                # validation, to produce ker_map_list(fake)
                if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                    save_img = opt['train'].get('save_img', False)
                    with_metrics = opt.get('metrics') is not None
                    if with_metrics:
                        metric_results_val = {metric: 0 for metric in opt['metrics'].keys()}
                    pbar = tqdm(total=val_size, unit='image')
                    for idx in range(val_size):
                        val_data = next(val_loader_iter)
                        img_name = osp.splitext(osp.basename(val_data['LQ_path'][0]))[0]
                        LQ, GT = val_data["LQ"], val_data["GT"]
                        LQ = util.upscale(LQ, scale)
                        noisy_state = sde.noise_state(LQ)  # 在LR上加噪声，得到噪声LR图，噪声是随机生成的

                        # valid Predictor
                        model.feed_data(noisy_state, LQ, GT)
                        model.test(sde)

                        if with_metrics:
                            # calculate metrics
                            for name, opt_ in opt['metrics'].items():
                                metric_data = dict(img1=model.output, img2=model.state_0)
                                metric_results_val[name] += calculate_metric(metric_data, opt_).item()

                        
                        visuals = model.get_current_visuals()
                        if eval_FID:
                            FID_dataloader.append((torch.unsqueeze(visuals['Output'], 0), torch.unsqueeze(visuals['GT'], 0)))

                        if save_img:
                            visuals = model.get_current_visuals()
                            output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                            gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                            # save the validation results
                            save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                            util.mkdirs(save_path)
                            save_name = save_path + '/'+ img_name + '.png'
                            util.save_img(output, save_name)

                        # calculate PSNR
                        # avg_psnr += util.calculate_psnr(util.tensor2img(visuals["Output"].squeeze()), util.tensor2img(visuals["GT"].squeeze()))

                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}.png')
                    pbar.close()

                    if with_metrics:
                        for metric in metric_results_val.keys():
                            metric_results_val[metric] /= (idx + 1)
                        
                        if opt.get('FID') is not None and len(FID_dataloader) > 2048:
                            logger.info('evaluating FID...')
                            metric_data = dict(data_generator = FID_dataloader)
                            metric_results_val['FID'] = calculate_metric(metric_data, opt['FID']).item()
                            FID_dataloader = []
                            logger.info('Done.')

                        log_str = f"Validation {opt['datasets']['val']['name']}\n"
                        for metric, value in metric_results_val.items():
                            log_str += f'\t # {metric}: {value:.4f}\n'
                        logger.info(log_str)
                        if tb_logger:
                            for metric, value in metric_results_val.items():
                                tb_logger.add_scalar(f'val/metrics/{metric}', value, current_step)

                    # avg_psnr = avg_psnr / idx

                    # if avg_psnr > best_psnr:
                    #     best_psnr = avg_psnr
                    #     best_iter = current_step

                    # # log
                    # logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                    # logger_val = logging.getLogger("val")  # validation logger
                    # logger_val.info(
                    #     "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                    #         epoch, current_step, avg_psnr
                    #     )
                    # )
                    # print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                    #         epoch, current_step, avg_psnr
                    #     ))
                    # # tensorboard logger
                    # if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    #     tb_logger.add_scalar("psnr", avg_psnr, current_step)
        
                if error.value:
                    sys.exit(0)
                #### save models and training states
                if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                    if rank <= 0:
                        logger.info("Saving models and training states.")
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
                
                train_data = prefetcher.next()

        if rank <= 0:
            logger.info("Saving the final model.")
            model.save("latest")
            logger.info("End of Predictor and Corrector training.")
        tb_logger.close()
    else:
        for epoch in range(start_epoch, total_epochs + 1):
            if opt["dist"]:
                train_sampler.set_epoch(epoch)
            for _, train_data in enumerate(train_loader):
                current_step += 1

                if current_step > total_iters:
                    break

                LQ, GT = train_data["LQ"], train_data["GT"]  #  b 3 32 32; b 3 128 128

                LQ = util.upscale(LQ, scale)  #  bicubic, which can be repleced by deep networks

                # random timestep and state (noisy map) via SDE
                timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)  # t=batchsize，states [b 3 128 128]

                model.feed_data(states, LQ, GT)  # xt, mu, x0, 将加了噪声的LR图xt，LR以及GT输入改进的UNet进行去噪

                model.optimize_parameters(current_step, timesteps, sde)  # 优化UNet

                model.update_learning_rate(
                    current_step, warmup_iter=opt["train"]["warmup_iter"]
                )

                if current_step % opt["logger"]["print_freq"] == 0:
                    logs = model.get_current_log_reset(sde, opt['logger']['print_freq'])
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

                # validation, to produce ker_map_list(fake)
                if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                    save_img = opt['train'].get('save_img', False)
                    with_metrics = opt.get('metrics') is not None
                    if with_metrics:
                        metric_results_val = {metric: 0 for metric in opt['metrics'].keys()}
                    pbar = tqdm(total=val_size, unit='image')
                    for idx in range(val_size):
                        val_data = next(val_loader_iter)
                        img_name = osp.splitext(osp.basename(val_data['LQ_path'][0]))[0]
                        LQ, GT = val_data["LQ"], val_data["GT"]
                        LQ = util.upscale(LQ, scale)
                        noisy_state = sde.noise_state(LQ)  # 在LR上加噪声，得到噪声LR图，噪声是随机生成的

                        # valid Predictor
                        model.feed_data(noisy_state, LQ, GT)
                        model.test(sde)

                        if with_metrics:
                            # calculate metrics
                            for name, opt_ in opt['metrics'].items():
                                metric_data = dict(img1=model.output, img2=model.state_0)
                                metric_results_val[name] += calculate_metric(metric_data, opt_).item()

                        
                        visuals = model.get_current_visuals()
                        if eval_FID:
                            FID_dataloader.append((torch.unsqueeze(visuals['Output'], 0), torch.unsqueeze(visuals['GT'], 0)))

                        if save_img:
                            visuals = model.get_current_visuals()
                            output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                            gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                            # save the validation results
                            save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                            util.mkdirs(save_path)
                            save_name = save_path + '/'+ img_name + '.png'
                            util.save_img(output, save_name)

                        # calculate PSNR
                        # avg_psnr += util.calculate_psnr(util.tensor2img(visuals["Output"].squeeze()), util.tensor2img(visuals["GT"].squeeze()))

                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}.png')
                    pbar.close()

                    if with_metrics:
                        for metric in metric_results_val.keys():
                            metric_results_val[metric] /= (idx + 1)
                        
                        if opt.get('FID') is not None and len(FID_dataloader) > 2048:
                            logger.info('evaluating FID...')
                            metric_data = dict(data_generator = FID_dataloader)
                            metric_results_val['FID'] = calculate_metric(metric_data, opt['FID']).item()
                            FID_dataloader = []
                            logger.info('Done.')

                        log_str = f"Validation {opt['datasets']['val']['name']}\n"
                        for metric, value in metric_results_val.items():
                            log_str += f'\t # {metric}: {value:.4f}\n'
                        logger.info(log_str)
                        if tb_logger:
                            for metric, value in metric_results_val.items():
                                tb_logger.add_scalar(f'val/metrics/{metric}', value, current_step)

                    # avg_psnr = avg_psnr / idx

                    # if avg_psnr > best_psnr:
                    #     best_psnr = avg_psnr
                    #     best_iter = current_step

                    # # log
                    # logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                    # logger_val = logging.getLogger("val")  # validation logger
                    # logger_val.info(
                    #     "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                    #         epoch, current_step, avg_psnr
                    #     )
                    # )
                    # print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                    #         epoch, current_step, avg_psnr
                    #     ))
                    # # tensorboard logger
                    # if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    #     tb_logger.add_scalar("psnr", avg_psnr, current_step)
        
                if error.value:
                    sys.exit(0)
                #### save models and training states
                if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                    if rank <= 0:
                        logger.info("Saving models and training states.")
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
            
        if rank <= 0:
            logger.info("Saving the final model.")
            model.save("latest")
            logger.info("End of Predictor and Corrector training.")
        tb_logger.close()


if __name__ == "__main__":
    main()
