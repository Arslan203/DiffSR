import datetime
import logging
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from .dist_util import get_dist_info, master_only
from basicsr.metrics import calculate_metric

initialized_logger = {}


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, ' f'iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k[2:]}', v, current_iter)
                elif k.startswith('m_'):
                    self.tb_logger.add_scalar(f'metrics/{k[2:]}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


class SamplesLogger():
    def __init__(self, opt, start_iter=1, tb_logger=None, **kwargs):
        self.exp_name = 'samples'
        self.interval = opt['logger']['samples']['samples_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

        train_loader, val_loader = kwargs.get('train_loader', None), kwargs.get('val_loader', None)
        train_samples = torch.randint(len(train_loader.dataset), (opt['logger']['samples']['samples_train'], ))
        test_samples = torch.randint(len(val_loader.dataset), (opt['logger']['samples']['samples_train'], ))
        
        train_samples_data = [train_loader.dataset[i] for i in train_samples]
        test_samples_data = [val_loader.dataset[i] for i in test_samples]


        self.samples_info = {'tr_samples':train_samples, 'tt_samples':test_samples,
                    'tr_images': torch.stack([sample['lq'] for sample in train_samples_data]),
                    'tt_images': torch.stack([sample['lq'] for sample in test_samples_data]),
                    'tr_masks': torch.stack([sample['gt'] for sample in train_samples_data]),
                    'tt_masks': torch.stack([sample['gt'] for sample in test_samples_data]),
                }


    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        current_iter = log_vars.pop('iter')
        epoch = log_vars.pop('epoch')

        message = (f'[{self.exp_name[:5]}][epoch:{epoch:3d}, ' f'iter:{current_iter:8,d}]')

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        logits = log_vars.pop('logits')
        masks = log_vars.pop('masks')
        metrics = log_vars.pop('metrics')

        samples_size = self.samples_info['tr_samples'].size(0)
        fig_tr, ax_tr = plt.subplots(samples_size, 3, figsize = (7, 7))

        metrics_eval = defaultdict(float)
        for sample in range(samples_size):
            image = self.samples_info['tr_images'][sample]
            mask = masks[sample]
            sr = logits[sample]
            
            # metrics_eval = [f'{metric_names[i]} = {round(metr(mask.unsqueeze(0), sr.unsqueeze(0)).item(), 3)}' for name, opt_ in metrics.items()]
            for name, opt_ in metrics.items():
                metrics_eval[name] += calculate_metric(dict(img1=mask.unsqueeze(0), img2=sr.unsqueeze(0)), opt_).item()
            image = image.permute(1, 2, 0).numpy()
            
            mask = mask.cpu().permute(1, 2, 0).numpy()
            
            sr = sr.cpu().permute(1, 2, 0).numpy()
            
            
            
            ax_tr[sample, 0].set_title('LR_train')
            ax_tr[sample, 1].set_title('HR_train')
            ax_tr[sample, 2].set_title('SR_train')

            ax_tr[sample, 0].imshow(image)
            ax_tr[sample, 1].imshow(mask)
            ax_tr[sample, 2].imshow(np.clip(sr, 0, 1))

            ax_tr[sample, 0].set_axis_off()
            ax_tr[sample, 1].set_axis_off()
            ax_tr[sample, 2].set_axis_off()
        
        metrics_eval = [f'{name} = {round(val / samples_size, 3)}' for name, val in metrics_eval.items()]
        fig_tr.suptitle('; '.join(metrics_eval))
        fig_tr.tight_layout()

        # writer.add_figure('train/samples', fig_tr, epoch)


        fig_tt, ax_tt = plt.subplots(samples_size, 3, figsize = (7, 7))
        metrics_eval = defaultdict(float)

        for sample in range(samples_size):
            image = self.samples_info['tt_images'][sample]
            mask = masks[samples_size + sample]
            sr = logits[samples_size + sample]
            
            for name, opt_ in metrics.items():
                metrics_eval[name] += calculate_metric(dict(img1=mask.unsqueeze(0), img2=sr.unsqueeze(0)), opt_).item()
            image = image.permute(1, 2, 0).numpy()
            
            mask = mask.cpu().permute(1, 2, 0).numpy()
            
            sr = sr.cpu().permute(1, 2, 0).numpy()
            
            
            ax_tt[sample, 0].set_title('LR_test')
            ax_tt[sample, 1].set_title('HR_test')
            ax_tt[sample, 2].set_title('SR_test')

            ax_tt[sample, 0].imshow(image)
            ax_tt[sample, 1].imshow(mask)
            ax_tt[sample, 2].imshow(np.clip(sr, 0, 1))

            ax_tt[sample, 0].set_axis_off()
            ax_tt[sample, 1].set_axis_off()
            ax_tt[sample, 2].set_axis_off()

        metrics_eval = [f'{name} = {round(val / samples_size, 3)}' for name, val in metrics_eval.items()]
        fig_tt.suptitle('; '.join(metrics_eval))
        fig_tt.tight_layout()

        # writer.add_figure('test/samples', fig_tt, epoch)
        if self.use_tb_logger and 'debug' not in self.exp_name:
            self.tb_logger.add_figure('train/samples', fig_tr, current_iter // self.interval)
            self.tb_logger.add_figure('val/samples', fig_tt, current_iter // self.interval)
        
        plt.close()

        self.logger.info(message)

@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=opt['name'], config=opt, project=project, sync_tensorboard=True)#, mode='offline')

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
