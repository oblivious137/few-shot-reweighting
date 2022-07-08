import xml.etree.ElementTree as ET
import os
import shutil
import json
import random

class2label = dict()
label2imgs = list()
img2box = dict()
few_class = ['scissors', 'toaster', 'cake', 'airplane',
    'pizza', 'kite', 'bench', 'hair drier', 'bicycle', 'bear']
shot_k = 10

f = open("coco/annotations/instances_train2017.json", "rb")
ins = json.load(f)
f.close()

clsinfo = ins['categories']
label2class = [x['name'] for x in clsinfo]
label2class.sort()
ol2nl = dict()
for i, c in enumerate(label2class):
    class2label[c] = i
    label2imgs.append(list())
for x in clsinfo:
    ol2nl[x['id']] = class2label[x['name']]

fid2name = dict()

for fil in ins['images']:
    fn = fil['file_name']
    if fil['coco_url'].find('train') < 0:
        continue
    fn = os.path.join("coco/images/train2017", fn)
    fid2name[fil['id']] = fn
    img2box[fn] = list()

for obj in ins['annotations']:
    if obj['iscrowd'] == 1 or obj['area'] < 1:
        continue
    if obj['image_id'] not in fid2name:
        continue
    fn = fid2name[obj['image_id']]
    label = ol2nl[obj['category_id']]
    tmp = obj['bbox']
    bbox = (tmp[0], tmp[1], tmp[0]+tmp[2], tmp[1]+tmp[3])
    label2imgs[label].append((bbox, fn))
    img2box[fn].append((label, bbox))

few_imgs = [list() for _ in range(len(class2label))]
for v in img2box.values():
    v.sort()

for i in range(len(few_imgs)):
    while len(few_imgs[i]) < shot_k:
        name = random.choice(label2imgs[i])[-1]
        if name in few_imgs[i]:
            continue
        box = img2box[name]
        fl = True
        objs = list(map(lambda x: x[0], box))
        for obj in objs:
            if objs.count(obj) >= 3:
                fl = False
                break
        if fl:
            for obj in objs:
                if len(few_imgs[obj]) < shot_k:
                    few_imgs[obj].append(name)

with open("coco/fewshot_imgs.json", "w") as f:
    json.dump(few_imgs, f)

with open("coco/label2class.json", "w") as f:
    json.dump(label2class, f)

with open("coco/label2imgs.json", "w") as f:
    json.dump(label2imgs, f)

with open("coco/class2label.json", "w") as f:
    json.dump(class2label, f)

with open("coco/img2box.json", "w") as f:
    json.dump(img2box, f)

with open("coco/fewshot_label.json", "w") as f:
    json.dump(list(map(lambda x: class2label[x], few_class)), f)
