import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['PairwiseContrastiveLoss']


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m

def split_pair(raw_tensor):
    bs = raw_tensor.size(0)
    resized_tensor = raw_tensor.view(bs//2, 2, -1).contiguous()
    split1, split2 = torch.split(resized_tensor, 1, 1)
    return split1.flatten(1), split2.flatten(1)

def gather_tensor(raw_tensor):
    tensor_large = [torch.zeros_like(raw_tensor) \
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_large, raw_tensor.contiguous())
    tensor_large = torch.cat(tensor_large, dim=0)
    return tensor_large

def calculate_dist(hidden1, hidden2, hidden1_large, hidden2_large, masks, loss_type, temp, q_feat=None, norm_base=None):

    norm_aa = norm_ab = norm_bb = norm_ba = 1
    if loss_type == 'rbe_loss':
        hidden1_norm = torch.linalg.norm(hidden1, dim=1)
        hidden2_norm = torch.linalg.norm(hidden2, dim=1)
        hidden1_large_norm = torch.linalg.norm(hidden1_large, dim=1)
        hidden2_large_norm = torch.linalg.norm(hidden2_large, dim=1)

        norm_ab = torch.matmul(hidden1_norm.unsqueeze(1), hidden2_large_norm.unsqueeze(0))
        norm_ba = torch.matmul(hidden2_norm.unsqueeze(1), hidden1_large_norm.unsqueeze(0))

    if 'contra' in loss_type or loss_type=='rbe_loss':
        
        logits_a = torch.matmul(hidden1, hidden2_large.permute(1, 0)) / norm_ab / temp
        logits_b = torch.matmul(hidden2, hidden1_large.permute(1, 0)) / norm_ba / temp

    elif loss_type=='triplet':

        logits_a = euclidean_dist(hidden1, hidden2_large)
        logits_b = euclidean_dist(hidden2, hidden1_large)

    return logits_a, logits_b

def rm_same_vid(logits_a, logits_b, vids, masks, batch_size, enlarged_batch_size, loss_type, q_vids=None):
    vid, _ = split_pair(vids.view(-1,1))
    vid_single = vid.expand(batch_size, enlarged_batch_size)
    vid_large = vid
    vid_large = vid_large.expand(enlarged_batch_size, batch_size).t().contiguous()
    vid_mask = (vid_single==vid_large).float()
    vid_mask = vid_mask * (1 - masks)
    # vid_mask = torch.cat((vid_mask, vid_mask), dim=1)
    if q_vids is not None:
        vid_single = vid.expand(batch_size, q_vids.size(0))
        vid_large = q_vids.view(-1,1).expand(q_vids.size(0), batch_size).t().contiguous()
        vid_mask_q = (vid_single==vid_large).float()
        vid_mask = torch.cat((vid_mask, vid_mask_q), dim=1)
    if 'contra' in loss_type or loss_type=='rbe_loss':
        logits_a = logits_a - vid_mask * 1e9
        logits_b = logits_b - vid_mask * 1e9
    elif loss_type=='triplet':
        logits_a = logits_a + vid_mask * 1e9
        logits_b = logits_b + vid_mask * 1e9
    return logits_a, logits_b


def calculate_loss(logits_a, logits_b, labels_idx, masks, loss_type, batch_size, criterion, hard_topk_neg, q_vids=None):
    if hard_topk_neg is not None:
        pos_a = torch.gather(logits_a, 1, labels_idx.view(-1, 1).cuda())
        pos_b = torch.gather(logits_b, 1, labels_idx.view(-1, 1).cuda())
        if 'contra' in loss_type or loss_type=='rbe_loss':
            if q_vids is None:
                neg_a = torch.topk(logits_a - masks * 1e9, hard_topk_neg, dim=1)[0]
                neg_b = torch.topk(logits_b - masks * 1e9, hard_topk_neg, dim=1)[0]
            else:
                neg_a = torch.topk(logits_a-torch.cat((masks, torch.zeros_like(masks), torch.zeros(batch_size, q_vids.size(0)).to(masks.device)),dim=1)*1e9, hard_topk_neg, dim=1)[0]
                neg_b = torch.topk(logits_b-torch.cat((masks, torch.zeros_like(masks), torch.zeros(batch_size, q_vids.size(0)).to(masks.device)),dim=1)*1e9, hard_topk_neg, dim=1)[0]
            hard_logits_a = torch.cat((pos_a, neg_a), dim=1)
            hard_logits_b = torch.cat((pos_b, neg_b), dim=1)
            hard_labels_idx = torch.zeros(batch_size).long()

            loss_a = criterion(hard_logits_a, hard_labels_idx.cuda())
            loss_b = criterion(hard_logits_b, hard_labels_idx.cuda())
        elif loss_type=='triplet':
            if (q_vids is None):
                neg_a = torch.topk(logits_a + masks * 1e9, hard_topk_neg, dim=1, largest=False)[0]
                neg_b = torch.topk(logits_b + masks * 1e9, hard_topk_neg, dim=1, largest=False)[0]
            else:
                neg_a = torch.topk(logits_a+torch.cat((masks, torch.zeros_like(masks), torch.zeros(batch_size, q_vids.size(0)).to(masks.device)),dim=1)*1e9, hard_topk_neg, dim=1, largest=False)[0]
                neg_b = torch.topk(logits_b+torch.cat((masks, torch.zeros_like(masks), torch.zeros(batch_size, q_vids.size(0)).to(masks.device)),dim=1)*1e9, hard_topk_neg, dim=1, largest=False)[0]
            pos_a = pos_a.expand_as(neg_a).contiguous().view(-1)
            pos_b = pos_b.expand_as(neg_b).contiguous().view(-1)
            neg_a, neg_b = neg_a.view(-1), neg_b.view(-1)
            hard_labels_idx = torch.ones_like(pos_a)
            loss_a = criterion(neg_a, pos_a, hard_labels_idx)
            loss_b = criterion(neg_b, pos_b, hard_labels_idx)
    else:
        loss_a = criterion(logits_a, labels_idx.cuda())
        loss_b = criterion(logits_b, labels_idx.cuda())
    return loss_a, loss_b

class PairwiseContrastiveLoss(nn.Module):

    def __init__(self, temp=0.1, margin=0.8, rm_duplicated=True, hard_topk_neg=None, loss_type='contra', queue=None, no_norm=False):
        super(PairwiseContrastiveLoss, self).__init__()
        self.temperature = temp
        self.rm_duplicated = rm_duplicated
        self.hard_topk_neg = hard_topk_neg

        if 'contra' in loss_type or loss_type=='rbe_loss':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type=='triplet':
            assert hard_topk_neg is not None, \
                   "Please select top-k negatives for triplet loss, by setting TRAIN.LOSS.HARD_TOPK"
            self.criterion = nn.MarginRankingLoss(margin=margin)
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))
        self.loss_type = loss_type
        self.queue = queue
        self.no_norm = no_norm

    def forward(self, inputs, vids):
        # features l2-norm
        if not self.no_norm:
            z = F.normalize(inputs, dim=1, p=2)
        else:
            z = inputs

        # split into pairs
        batch_size = z.size(0)//2
        hidden1, hidden2 = split_pair(z)

        hidden1_large = hidden1
        hidden2_large = hidden2
        enlarged_batch_size = hidden1_large.size(0)

        # create label
        labels_idx = torch.arange(batch_size)
        masks = torch.zeros(batch_size, enlarged_batch_size).scatter_(1, labels_idx.unsqueeze(1), 1).cuda()

        # calculate logits
        logits_a, logits_b = calculate_dist(hidden1, hidden2, hidden1_large, hidden2_large, masks, self.loss_type, self.temperature)

        # remove duplicated vid
        if self.rm_duplicated:
            logits_a, logits_b = rm_same_vid(logits_a, logits_b, vids, masks, batch_size, enlarged_batch_size, self.loss_type)

        # compute loss
        loss_a, loss_b = calculate_loss(logits_a, logits_b, labels_idx, masks, self.loss_type, batch_size, self.criterion, self.hard_topk_neg)
        return loss_a + loss_b

class PairwiseCompatibleLoss(PairwiseContrastiveLoss):

    def forward(self, inputs, inputs_old, targets, vids):
        if not self.no_norm:
            # features l2-norm
            z = F.normalize(inputs, dim=1, p=2)
            z_old = F.normalize(inputs_old, dim=1, p=2).detach()
        else:
            z = inputs
            z_old = inputs_old.detach()

        # split into pairs
        batch_size = z.size(0)//2
        hidden1, hidden2 = split_pair(z)
        hidden1_old, hidden2_old = split_pair(z_old)

        # gather from all gpus
        hidden1_large = gather_tensor(hidden1)
        hidden2_large = gather_tensor(hidden2)
        hidden1_old_large = gather_tensor(hidden1_old)
        hidden2_old_large = gather_tensor(hidden2_old)

        enlarged_batch_size = hidden1_large.size(0)

        # create label
        labels_idx = torch.arange(batch_size) + dist.get_rank() * batch_size
        masks = torch.zeros(batch_size, enlarged_batch_size).scatter_(1, labels_idx.unsqueeze(1), 1).cuda()

        # calculate logits
        logits_a, logits_b = calculate_dist(hidden1, hidden2, hidden1_old_large, hidden2_old_large, masks, self.loss_type, self.temperature)

        # remove duplicated vid
        if self.rm_duplicated:
            logits_a, logits_b = rm_same_vid(logits_a, logits_b, vids, masks, batch_size, enlarged_batch_size, self.loss_type)

        # compute loss
        loss_a, loss_b = calculate_loss(logits_a, logits_b, labels_idx, masks, self.loss_type, batch_size, self.criterion, self.hard_topk_neg)
        return loss_a + loss_b
