import numpy as np

fine_tune = True
gpu = True
pretrain_darknet = "pretrained/darknet19_448.conv.23"
continue_train = False or fine_tune
model_dir = "models/log_wh"

print_freq = 64
batch_size = 1


dataroot = "coco"
anchors = ((0.57273, 0.677385),
           (1.87446, 2.06253),
           (3.33843, 5.47434),
           (7.88282, 3.52778),
           (9.77052, 9.16828))
reweight_size = 416
ml = 0
mr = 0

# dataroot = "dataset"
# anchors = (
#     (1.3221, 1.73145),
#     (3.19275, 4.00944),
#     (5.05587, 8.09892),
#     (9.47112, 4.84053),
#     (11.2364, 10.0071)
# )
# reweight_size = 416
# ml = 3
# mr = 5

if fine_tune:
    print("fine tune ...")
    model_dir = "models/finetune"
    lr_decay_epochs = []
    lr_decay = []
    save_freq = 20
    init_learning_rate = 1e-3 / batch_size
    max_epoch = 4000
    neg = 0.0
    neg_lr_scale = 1.5
else:
    lr_decay_epochs = [2, 60, 90]
    lr_decay = [10, 0.1, 0.1]
    save_freq = 3
    init_learning_rate = 1e-4 / batch_size
    max_epoch = 120
    neg = 1.0
    neg_lr_scale = 3.0

last_epoch = -1

weight_decay = 0.0005
momentum = 0.9

model_dir = model_dir + ("_coco" if dataroot=="coco" else "_voc")

num_anchors = len(anchors)

multi_scale_in = list()
multi_scale_out = list()
ml = reweight_size//32 - ml
mr = reweight_size//32 + mr
for i in range(ml, mr + 1):
    multi_scale_in.append([i*32, i*32])
    multi_scale_out.append([i, i])

lambda_class = 1
lambda_obj = 5
lambda_noobj = 1
lambda_coord = 1
iou_thresh = 0.6
dynamic = True
debug = False
