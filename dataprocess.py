import xml.etree.ElementTree as ET
import os
import shutil
import json
import random

datas = os.listdir("dataset")
label2class = list()
class2label = dict()
label2imgs = list()
img2box = dict()
few_class = ["bird", "bus", "cow", "motorbike", "sofa"]
shot_k = 10

for root in datas:
    if root.find("test")>=0 or not os.path.isdir(os.path.join("dataset", root)):
        continue
    for f in os.listdir(os.path.join("dataset", root, "Annotations")):
        tree = ET.parse(os.path.join("dataset", root, "Annotations", f))
        treeroot = tree.getroot()
        fn = treeroot.find("filename").text
        boxes = list()
        for obj in treeroot.findall("object"):
            difficult = obj.find("difficult").text
            if int(difficult) == 1:
                continue
            clas = obj.find("name").text
            if clas not in class2label:
                label2imgs.append(list())
                class2label[clas] = len(class2label)
                label2class.append(clas)
            label = class2label[clas]
            bbox = obj.find("bndbox")
            min0, min1, max0, max1 = [bbox.find("ymin"), bbox.find("xmin"), bbox.find("ymax"), bbox.find("xmax")]
            min0, min1, max0, max1 = list(map(lambda x: float(x.text), (min0, min1, max0, max1)))
            boxes.append((label, (min1, min0, max1, max0)))
            label2imgs[label].append(((min1, min0, max1, max0), os.path.join("dataset", root, "JPEGImages", fn)))
        img2box[os.path.join("dataset", root, "JPEGImages", fn)] = sorted(boxes)

few_imgs=[list() for _ in range(len(class2label))]


for i in range(len(few_imgs)):
    while len(few_imgs[i]) < shot_k:
        name = random.choice(label2imgs[i])[-1]
        if name in few_imgs[i]:
            continue
        box = img2box[name]
        fl = True
        objs = list(map(lambda x: x[0], box))
        for obj in objs:
            if len(few_imgs[obj]) > shot_k or objs.count(obj) >= 3:
                fl = False
                break
        if fl:
            for obj in objs:
                few_imgs[obj].append(name)

with open("dataset/fewshot_imgs.json", "w") as f:
    json.dump(few_imgs, f)

with open("dataset/label2class.json", "w") as f:
    json.dump(label2class, f)

with open("dataset/label2imgs.json", "w") as f:
    json.dump(label2imgs, f)

with open("dataset/class2label.json", "w") as f:
    json.dump(class2label, f)

with open("dataset/img2box.json", "w") as f:
    json.dump(img2box, f)

with open("dataset/fewshot_label.json", "w") as f:
    json.dump(list(map(lambda x: class2label[x], few_class)), f)
