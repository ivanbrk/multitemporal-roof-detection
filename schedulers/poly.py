from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_steps, power=0.9, min_lr=1e-6, last_epoch=-1):
        self.max_steps = max(1, int(max_steps))
        self.power = float(power)
        self.min_lr = float(min_lr)
        super(PolyLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        current_step = min(max(self.last_epoch, 0), self.max_steps)
        coefficient = (1.0 - float(current_step) / float(self.max_steps)) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * coefficient for base_lr in self.base_lrs
        ]


def build_scheduler(name, optimizer, total_steps, power=0.9, min_lr=1e-6):
    normalized = str(name).lower()
    if normalized != "poly":
        raise ValueError("Unsupported scheduler: %s" % name)
    return PolyLR(optimizer=optimizer, max_steps=total_steps, power=power, min_lr=min_lr)
