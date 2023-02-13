import torch

# https://zhuanlan.zhihu.com/p/72956595
# https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256

# create stream with:
# stm = torch.cuda.Stream()


class data_prefetcher():
    def __init__(self, loader, stm):
        self.loader = iter(loader)
        self.stream = stm
        self.preload()

    def preload(self):
        try:
            # self.next_input, self.next_target, self.next_vid = next(self.loader)
            self.next_data = next(self.loader)
        except StopIteration:
            # self.next_input = None
            # self.next_target = None
            # self.next_vid = None
            self.next_data = None
            return None

        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # self.next_vid = self.next_vid.cuda(non_blocking=True)
            # self.next_input = self.next_input.float()
            if type(self.next_data) == torch.Tensor:
                self.next_data = [self.next_data]
            
            for i in range(len(self.next_data)):
                self.next_data[i] = self.next_data[i].cuda(non_blocking=True)
        return True

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        # input = self.next_input
        # target = self.next_target
        # vid = self.next_vid
        data = self.next_data
        flag = self.preload()
        return (*data, flag)

    def __delete__(self, instance):
        # del self.next_input, self.next_target, self.next_vid
        del self.next_data
