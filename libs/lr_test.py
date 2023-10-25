import torch
import matplotlib.pyplot as plt

from sgdr_restarts_warmup import CosineAnnealingWarmupRestarts

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
use_warmup = True
steps_per_epoch_train = 512
lr_sched = CosineAnnealingWarmupRestarts(optimizer,
                                         first_cycle_steps=14 * steps_per_epoch_train,
                                         cycle_mult=1.5,
                                         max_lr=1e-4,
                                         min_lr=5e-7,
                                         warmup_steps=steps_per_epoch_train if use_warmup else 0,
                                         gamma=0.8,
                                         last_epoch=-1)

lrs = []
min_lr = 5e-7
for i in range(512*64):
    lr_sched.step()
    if min_lr == optimizer.param_groups[0]["lr"]:
        print('!', i)
    lrs.append(
        optimizer.param_groups[0]["lr"]
    )

plt.plot(lrs)
plt.savefig('./lr_figure')
