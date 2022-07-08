gpu=True
dynamic=True
nms_threshold = 0.45
iou_threshold = 0.005
model_path = "models/finetune/model_1999.pkl"
anchors = (
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071)
)
num_anchors = len(anchors)
reweight_size = 416
image_size = 416
