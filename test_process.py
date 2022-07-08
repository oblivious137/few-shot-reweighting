import xml.etree.ElementTree as ET
import os
import shutil
import json, pickle
import random

datas = os.listdir("dataset")
few_class = ["bird", "bus", "cow", "motorbike", "sofa"]
shot_k = 10

with open("dataset/label2class.json", "r") as f:
    label2class = json.load(f)
with open("dataset/class2label.json", "r") as f:
    class2label = json.load(f)
gt = dict()

root = "VOC2007test"
for f in os.listdir(os.path.join("dataset", root, "Annotations")):
    tree = ET.parse(os.path.join("dataset", root, "Annotations", f))
    treeroot = tree.getroot()
    fn = treeroot.find("filename").text
    boxes = list()
    for obj in treeroot.findall("object"):
        clas = obj.find("name").text
        diff = int(obj.find("difficult").text)
        label = class2label[clas]
        bbox = obj.find("bndbox")
        min0, min1, max0, max1 = [bbox.find("ymin"), bbox.find("xmin"), bbox.find("ymax"), bbox.find("xmax")]
        min0, min1, max0, max1 = list(map(lambda x: float(x.text), (min0, min1, max0, max1)))
        if (fn, label) not in gt:
            gt[(fn, label)] = list()
        gt[(fn, label)].append((min0, min1, max0, max1))

with open("test_gt.pkl", "wb") as f:
    pickle.dump(gt, f)
