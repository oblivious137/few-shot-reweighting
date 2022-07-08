gpu=True
dynamic=True
nms_threshold = 0.45
iou_threshold = 0.005
model_path = "models/log_wh_coco/model_1999.pkl"
anchors = ((0.57273, 0.677385),
           (1.87446, 2.06253),
           (3.33843, 5.47434),
           (7.88282, 3.52778),
           (9.77052, 9.16828))
reweight_size = 416
num_anchors = len(anchors)
reweight_size = 416
image_size = 416
