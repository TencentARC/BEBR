import os
import numpy as np
import ast
from tqdm import tqdm

import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, logger, optimizer=None, lr_scheduler=None, state_dict_key='model'):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if (state_dict_key is not None):
        msg = model.load_state_dict(checkpoint[state_dict_key], strict=False)
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and optimizer is not None and lr_scheduler is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, model_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if model_ema is not None:
        save_state['model_ema'] = model_ema.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_feature(path):
    if path.endswith('.npy'):
        return np.load(path).astype('float32')
    
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for d in tqdm(lines):
        data.append(ast.literal_eval(d.strip()))
    
    feat =  np.array(data).astype('float32')
    basename = os.path.basename(path).split('.')[0] + '.npy'
    dirname = os.path.dirname(path)
    np.save(os.path.join(dirname, basename), feat)
    
    return feat


def load_img_path(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split()[0] for line in lines]

def merge_feat_npy(feat_dir, prefix, num):
    feat_all = np.load(os.path.join(feat_dir, '{}_0.npy'.format(prefix)))
    for i in range(1, num):
        feat_part = np.load(os.path.join(feat_dir, '{}_{}.npy'.format(prefix, i)))
        feat_all = np.concatenate((feat_all, feat_part), axis=0)
    
    return feat_all
