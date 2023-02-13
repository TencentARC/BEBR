import os
import time
import argparse
import datetime
import numpy as np
import glob
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from timm.utils import accuracy, AverageMeter

from vfp.config import get_config
from vfp.models import build_binary_head
from vfp.data import build_loader, data_prefetcher
from vfp.lr_scheduler import build_scheduler
from vfp.optimizer import build_optimizer
from vfp.logger import create_logger
from vfp.utils import load_checkpoint, save_checkpoint, get_grad_norm, \
            auto_resume_helper
from vfp.loss import PairwiseContrastiveLoss
from vfp.queue import Queue

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Binary training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    # dataset
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')

    # training config
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--resume-key', default='model', help='resume checkpoint key')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    data_loader_query, data_loader_gallery = build_loader(config, is_train=False)
    # build dataset
    data_loader_train, _ = build_loader(config, is_train=True)

    logger.info(f"Creating binary transformation:{config.BINARY.TRANS.TYPE}")
    model, num_features = build_binary_head(config.BINARY, config.BINARY.TRANS.RBE.INPUT_DIM)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    model.cuda()
    logger.info(str(model))

    # build optimizer
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer)

    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            max_accuracy = load_checkpoint(config, model, logger, optimizer, lr_scheduler, state_dict_key='model')
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    stm = torch.cuda.Stream()

    # build loss
    if (config.TRAIN.LOSS.NAME=='pairwise_contrastive'):
        criterion = PairwiseContrastiveLoss(temp=config.TRAIN.LOSS.TEMP,
                                            margin=config.TRAIN.LOSS.MARGIN,
                                            rm_duplicated=config.TRAIN.LOSS.RM_DUP,
                                            hard_topk_neg=config.TRAIN.LOSS.HARD_TOPK,
                                            loss_type=config.TRAIN.LOSS.TYPE,
                                            no_norm=True).cuda()
        criterion = {'base': criterion}
    else:
        raise NotImplementedError("Unknown loss type.")

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(stm, config, model, criterion, data_loader_train, optimizer, epoch)
        
        if (epoch+1) % config.SAVE_FREQ == 0 or (epoch+1) == config.TRAIN.EPOCHS:
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        if (epoch+1) % config.EVAL_FREQ == 0:
            logger.info(f"==============> Start testing model....................")
            extract_features(stm, config, data_loader_query, model, 'image')
            extract_features(stm, config, data_loader_gallery, model, 'txt')

            run_eval(config)

        lr_scheduler.step(epoch + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(stm, config, model, criterion, data_loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    prefetcher = data_prefetcher(data_loader, stm)
    samples, targets, flag = prefetcher.next()
    idx = -1
    while samples is not None:
        idx += 1
        binary_feat = model(samples)

        loss = criterion['base'](binary_feat, targets)

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        if not flag:
            break
        samples, targets, flag = prefetcher.next()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    del prefetcher


@torch.no_grad()
def extract_features(stm, config, data_loader, model, tag):
    model.eval()

    batch_time = AverageMeter()

    feat_size = len(data_loader) * config.DATA.BATCH_SIZE
    feat_dim = config.BINARY.TRANS.RBE.OUTPUT_DIM
    feat_all = np.zeros((feat_size, feat_dim), dtype=np.float32)

    end = time.time()
    prefetcher = data_prefetcher(data_loader, stm)
    samples, flag = prefetcher.next()
    idx = -1
    cnt = 0
    while samples is not None:
        idx += 1
        output = model(samples)
        output = output.cpu().data.squeeze().numpy()

        feat_all[cnt : (cnt + output.shape[0])] = output
        cnt += output.shape[0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

        if not flag:
            break
        samples, flag = prefetcher.next()

    del prefetcher
    np.save(os.path.join(config.OUTPUT, f'feat_{tag}.npy'), feat_all[:cnt])

def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = np.zeros((max_pairs, max_pairs), dtype=np.float32)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ np.transpose(batch_caption_emb)
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict

def run_eval(config):
    image_feat = np.load(os.path.join(config.OUTPUT, 'feat_image.npy'))
    txt_feat = np.load(os.path.join(config.OUTPUT, 'feat_txt.npy'))

    image_feat = image_feat / np.linalg.norm(image_feat,axis=1)[:,np.newaxis]
    txt_feat = txt_feat / np.linalg.norm(txt_feat,axis=1)[:,np.newaxis]

    similarity_scores = compute_similarity(image_feat, txt_feat)
    i2t_dict = compute_retrieval(similarity_scores)
    logger.info('i2t {}'.format(i2t_dict))


if __name__ == '__main__':
    _, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
