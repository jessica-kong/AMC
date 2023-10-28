class WarmupLR:
    def __init__(self, optimizer, warmup_steps, init_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.last_epoch = last_epoch
        self.step_num = 0

    def get_lr(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            return self.init_lr * self.step_num / self.warmup_steps
        else:
            return self.init_lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
