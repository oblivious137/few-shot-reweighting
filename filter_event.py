from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

ea=event_accumulator.EventAccumulator("models/log_wh/log/events.out.tfevents.1606789066.linchpin-OMEN-Laptop.5127.0")
ea.Reload()
writer = SummaryWriter(log_dir="tmp")

iteration = int(input('the trunc iter: '))

for scalar in ea.scalars.Keys():
    for i in ea.scalars.Items(scalar):
        if i.step <= iteration:
            writer.add_scalar(scalar, i.value, global_step=i.step)
        else:
            print(i.step)
writer.close()