import torch
import torch.nn as nn


class Queue(nn.Module):
    def __init__(self, dim, K=65536):
        super(Queue, self).__init__()
        self.register_buffer("features", torch.randn(K, dim))
        self.register_buffer("vids", torch.randn(K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, vids):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        vids = concat_all_gather(vids)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.features[ptr:ptr + batch_size] = keys
        self.vids[ptr:ptr + batch_size] = vids
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def reset(self):
        self.queue_ptr[0] = 0

    def get(self):
        ptr = int(self.queue_ptr)
        if (ptr==0):
            return None, None
        fea = self.features[:ptr].clone().detach()
        lab = self.vids[:ptr].clone().detach()
        return fea, lab


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
