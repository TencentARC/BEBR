import os
import time
import argparse
import datetime
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter, ModelEmaV2

from vfp.config import get_config
from vfp.models import build_binary_head
from vfp.data import build_loader, data_prefetcher
from vfp.lr_scheduler import build_scheduler
from vfp.optimizer import build_optimizer
from vfp.logger import create_logger
from vfp.utils import load_checkpoint, save_checkpoint, get_grad_norm, \
            auto_resume_helper, reduce_tensor, load_feature, load_img_path, merge_feat_npy
from vfp.loss import PairwiseCompatibleLoss, PairwiseContrastiveLoss
from vfp.queue import Queue

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Compatible training script', add_help=False)
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

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    # build dataset
    dataset_train, data_loader_train = build_loader(config, is_train=True)
    dataset_query, dataset_gallery, data_loader_query, data_loader_gallery = build_loader(config, is_train=False)

    # build model
    logger.info(f"Creating binary transformation:{config.BINARY.TRANS.TYPE}")
    model, num_features = build_binary_head(config.BINARY, config.BINARY.TRANS.RBE.INPUT_DIM)
    if config.BINARY.MODEL.RESUME:
        load_checkpoint(config.BINARY, model, logger, state_dict_key=config.BINARY.MODEL.RESUME_KEY)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    model.cuda()
    logger.info(str(model))

    # build optimizer
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = 0.0

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_ema = None
    queue = {}
    queue['old'] = None
    queue['new'] = None
    if config.TRAIN.LOSS.QUEUE>0:
        # assert model_ema is not None, "model_ema is required for queue"
        assert config.TRAIN.LOSS.QUEUE % (config.DATA.BATCH_SIZE * dist.get_world_size()) == 0
        queue['old'] = Queue(num_features, K=config.TRAIN.LOSS.QUEUE)
        queue['new'] = Queue(num_features, K=config.TRAIN.LOSS.QUEUE)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            max_accuracy = load_checkpoint(config, model.module, logger, optimizer, lr_scheduler, state_dict_key='model')
            if model_ema is not None:
                load_checkpoint(config, model_ema.module.module, logger, state_dict_key='model_ema')
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    stm = torch.cuda.Stream()

    no_norm = True
    if config.TRAIN.LOSS.TYPE == 'rbe_loss':
        no_norm = False
    # build loss
    criterion = {}
    criterion['base'] = PairwiseContrastiveLoss(temp=config.TRAIN.LOSS.TEMP,
                                                margin=config.TRAIN.LOSS.MARGIN,
                                                rm_duplicated=config.TRAIN.LOSS.RM_DUP,
                                                hard_topk_neg=config.TRAIN.LOSS.HARD_TOPK,
                                                loss_type=config.TRAIN.LOSS.TYPE,
                                                queue=queue['new'],
                                                no_norm=no_norm).cuda()
    if config.COMPATIBLE.ACTIVATE:
        criterion['bct'] = PairwiseCompatibleLoss(temp=config.TRAIN.LOSS.TEMP,
                                                margin=config.TRAIN.LOSS.MARGIN,
                                                rm_duplicated=config.TRAIN.LOSS.RM_DUP,
                                                hard_topk_neg=config.TRAIN.LOSS.HARD_TOPK,
                                                loss_type=config.TRAIN.LOSS.TYPE,
                                                queue=queue['old'],
                                                no_norm=no_norm).cuda()

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if epoch == 1 and config.MODEL.EMA:
            model_ema = ModelEmaV2(model, decay=config.MODEL.EMA_DECAY)
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(stm, config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, model_ema=model_ema, queue=queue)
        if dist.get_rank() == 0 and ((epoch+1) % config.SAVE_FREQ == 0 or (epoch+1) == config.TRAIN.EPOCHS):
            if model_ema is not None:
                save_checkpoint(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema.module.module)
            else:
                save_checkpoint(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger)

        if ((epoch+1) % config.EVAL_FREQ == 0 or (epoch+1) == config.TRAIN.EPOCHS):
            logger.info(f"==============> Start testing model....................")
            extract_features(stm, config, data_loader_query, model, 'q')
            extract_features(stm, config, data_loader_gallery, model, 'db')
            dist.barrier()

            acc_bc, acc_new= 0.0, 0.0
            if (dist.get_rank() == 0):
                acc_bc, acc_new = faiss_search(config)
            dist.barrier()

            acc_bc_ema, acc_ema_new = 0.0, 0.0
            if model_ema is not None:
                logger.info(f"==============> Start testing ema model....................")
                extract_features(stm, config, data_loader_query, model_ema.module, 'q_ema')
                extract_features(stm, config, data_loader_gallery, model_ema.module, 'db_ema')
                dist.barrier()

                if (dist.get_rank() == 0):
                    acc_bc_ema, acc_ema_new = faiss_search(config, tag='_ema')

            logger.info(f' * [Epoch {epoch}] Top-{config.TEST.TOP_K} New PR: {acc_new:.2f}%, Backward PR: {acc_bc:.2f}% (EMA New PR: {acc_ema_new:.2f}%, Backward PR: {acc_bc_ema:.2f}%)')

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(stm, config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, model_ema=None, queue=None):
    model.module.train()
    optimizer.zero_grad()
    if queue is not None and queue['new'] is not None and queue['old'] is not None:
        queue['new'].reset()
        queue['old'].reset()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    prefetcher = data_prefetcher(data_loader, stm)
    samples, samples_old, targets, vids, flag = prefetcher.next()
    idx = -1
    while samples is not None:
        idx += 1
        new_feat = model(samples)
        old_feat = samples_old
        if (new_feat.size(1)>old_feat.size(1)):
            new_feat_bct = torch.nn.functional.normalize(new_feat, p=2, dim=1)[:, :old_feat.size(1)]
        else:
            new_feat_bct = new_feat

        if epoch > 0 and model_ema is not None and queue['new'] is not None and queue['old'] is not None:
            with torch.no_grad():
                new_feat_ema = model_ema.module(samples)
            loss = criterion['base'].forward_with_queue(new_feat, new_feat_ema, targets, vids)
            queue['new']._dequeue_and_enqueue(new_feat_ema, vids)
        else:
            loss = criterion['base'](new_feat, targets, vids)

        if config.COMPATIBLE.ACTIVATE:
            if epoch > 0 and model_ema is not None and queue['new'] is not None and queue['old'] is not None:
                loss += criterion['bct'].forward_with_queue(new_feat_bct, old_feat, targets, vids)
                queue['old']._dequeue_and_enqueue(old_feat, vids)
            else:
                loss += criterion['bct'](new_feat_bct, old_feat, targets, vids)

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

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
        samples, samples_old, targets, vids, flag = prefetcher.next()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    del prefetcher


@torch.no_grad()
def extract_features(stm, config, data_loader, model, tag):
    model.module.eval()
    rank = dist.get_rank()

    batch_time = AverageMeter()
    # feat_file = open(os.path.join(config.OUTPUT, f'feat_{tag}_new_{rank}.txt'), 'w')
    # feat_file_old = open(os.path.join(config.OUTPUT, f'feat_{tag}_old_{rank}.txt'), 'w')

    feat_size = len(data_loader) * config.DATA.BATCH_SIZE
    feat_dim = config.BINARY.TRANS.RBE.OUTPUT_DIM
    new_feat_all = np.zeros((feat_size, feat_dim), dtype=np.float32)
    old_feat_all = np.zeros((feat_size, feat_dim), dtype=np.float32)

    end = time.time()
    prefetcher = data_prefetcher(data_loader, stm)
    samples, samples_old, targets, _, flag = prefetcher.next()
    idx = -1
    cnt = 0
    while samples is not None:
        idx += 1
        new_feat = model(samples)
        # new_feat = new_feat[:, :old_feat.size(1)]

        # old_feat = old_feat.cpu().data.squeeze().numpy()
        # for item in old_feat:
        #     feat_file_old.write("{}\n".format(list(item)))

        new_feat = new_feat.cpu().data.numpy()
        old_feat = samples_old.cpu().data.numpy()
        # for item in new_feat:
        #     feat_file.write("{}\n".format(list(item)))

        new_feat_all[cnt : (cnt + new_feat.shape[0])] = new_feat
        old_feat_all[cnt : (cnt + old_feat.shape[0])] = old_feat
        cnt += new_feat.shape[0]

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
        samples, samples_old, targets, _, flag = prefetcher.next()

    del prefetcher
    np.save(os.path.join(config.OUTPUT, f'feat_{tag}_new_{rank}.npy'), new_feat_all[:cnt])
    np.save(os.path.join(config.OUTPUT, f'feat_{tag}_old_{rank}.npy'), old_feat_all[:cnt])
    # feat_file.close()
    # if (feat_file_old is not None):
    #     feat_file_old.close()


def faiss_search(config, tag=''):
    # merge txt
    gpu_num = dist.get_world_size()

    q_feat = merge_feat_npy(config.OUTPUT, f'feat_q{tag}_new', gpu_num)
    db_feat = merge_feat_npy(config.OUTPUT, f'feat_db{tag}_new', gpu_num)

    q_feat_old = merge_feat_npy(config.OUTPUT, f'feat_q{tag}_old', gpu_num)
    db_feat_old = merge_feat_npy(config.OUTPUT, f'feat_db{tag}_old', gpu_num)

    np.save(os.path.join(config.OUTPUT, f'feat_q{tag}_new.npy'), q_feat)
    np.save(os.path.join(config.OUTPUT, f'feat_db{tag}_new.npy'), db_feat)
    np.save(os.path.join(config.OUTPUT, f'feat_q{tag}_old.npy'), q_feat_old)
    np.save(os.path.join(config.OUTPUT, f'feat_db{tag}_old.npy'), db_feat_old)

    os.system("rm {}".format(os.path.join(config.OUTPUT, f'feat_q{tag}_new_{{0..{gpu_num-1}}}.npy')))
    os.system("rm {}".format(os.path.join(config.OUTPUT, f'feat_db{tag}_new_{{0..{gpu_num-1}}}.npy')))
    os.system("rm {}".format(os.path.join(config.OUTPUT, f'feat_q{tag}_old_{{0..{gpu_num-1}}}.npy')))
    os.system("rm {}".format(os.path.join(config.OUTPUT, f'feat_db{tag}_old_{{0..{gpu_num-1}}}.npy')))

    QueryImgPath = load_img_path(config.DATA.QUERY_LIST)
    DbImgPath = load_img_path(config.DATA.GALLERY_LIST)

    if (config.TEST.DIS_METRIC == 'cos'):
        db_feat = db_feat / np.linalg.norm(db_feat,axis=1)[:,np.newaxis]
        db_feat_old = db_feat_old / np.linalg.norm(db_feat_old,axis=1)[:,np.newaxis]
        q_feat = q_feat / np.linalg.norm(q_feat,axis=1)[:,np.newaxis]
        q_feat_old = q_feat_old / np.linalg.norm(q_feat_old,axis=1)[:,np.newaxis]

    def build_index(dim):
        if (config.TEST.DIS_METRIC == 'cos'):
            index = faiss.IndexFlatIP(dim)
        elif (config.TEST.DIS_METRIC == 'l2'):
            index = faiss.IndexFlatL2(dim)
        else:
            raise NotImplementedError("Unknown distance metric: {}".format(config.TEST.DIS_METRIC))
        return index

    def eval(sort_index, fqueryImg, fdbImg):
        retrieval = 0
        for row, LIndex in enumerate(sort_index):
            queryImgName = fqueryImg[row].split('/')[-1]
            query_vid = queryImgName.split('_')[0]
            count = 0
            for item in LIndex:
                dbImgName = fdbImg[item].split('/')[-1]
                db_vid = dbImgName.split('_')[0]
                if query_vid == db_vid:
                    count += 1
            retrieval += count
        return retrieval
    acc_bc = 0.0

    if config.COMPATIBLE.ACTIVATE:
    # test backward compatibility
        dim = db_feat_old.shape[1]
        index = build_index(dim)
        index.add(db_feat_old)
        if (q_feat.shape[1] > dim):
            q_feat_cut = q_feat[:,:dim].copy(order='C')
            if (config.TEST.DIS_METRIC == 'cos'):
                q_feat_cut = q_feat_cut / np.linalg.norm(q_feat_cut,axis=1)[:,np.newaxis]
        elif (q_feat.shape[1] == dim):
            q_feat_cut = q_feat
        else:
            raise "Query dimension is smaller than gallery dimenson"
        _, I = index.search(q_feat_cut, config.TEST.TOP_K)
        retrival = eval(I, QueryImgPath, DbImgPath)
        acc_bc = float(retrival) / len(DbImgPath) * 100.

    # test new model
    dim = db_feat.shape[1]
    index = build_index(dim)
    index.add(db_feat)
    _, I = index.search(q_feat, config.TEST.TOP_K)
    retrival = eval(I, QueryImgPath, DbImgPath)
    acc_new = float(retrival) / len(DbImgPath) * 100.

    return acc_bc, acc_new


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_address = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    print(f"MASTER_ADDR and MASTER_PORT in environ: {master_address}:{master_port}")
    config.defrost()
    config.LOCAL_RANK = rank % torch.cuda.device_count()
    config.freeze()
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend="nccl",
                                        init_method='tcp://{}:{}'.format(master_address, master_port),
                                        rank=rank,
                                        world_size=world_size
                                )
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
