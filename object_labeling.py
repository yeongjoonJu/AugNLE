import argparse, os
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests
import torch
import cv2
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
import urllib.request
from collections import defaultdict
import re
import jsonlines

# pip install jsonlines


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(img, prob, boxes, id2label):
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        (xmin, ymin, xmax, ymax) = list(map(int, (xmin, ymin, xmax, ymax)))
        img = cv2.rectangle(img, (abs(xmin), abs(ymin)), (xmax, ymax), color=c, thickness=3)
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        cv2.putText(img, text, (xmin, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imwrite("test.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

number2text = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
}

label2plural = {
    "person": "people", "sheep": "sheep", "knife": "knives"
}

def get_detected_objs(probs, id2label):
    objs = []
    for p in probs:
        cl = p.argmax()
        objs.append(id2label[cl.item()])
    
    objs = Counter(objs)

    detected_str = []
    detected_obj_cnt = 0
    for word, cnt in objs.items():
        detected_obj_cnt += cnt
        if cnt >= 2:
            if cnt > 9:
                cnt = "more than nine"
            else:
                cnt = number2text[cnt]
            if word in label2plural:
                word = label2plural[word]
            else:
                word = word+" -s"
            word = f"{cnt} {word}"
        else:
            word = f"one {word}"

        detected_str.append(word)
    
    prefix = "there are " if detected_obj_cnt > 1 else "there is "

    return prefix + ", ".join(detected_str) if detected_obj_cnt > 0 else ""


def VQA_X_obj_labeling(model, id2label, args):
    """
    image_dir: images/train2014 | images/val2014
    anno_path: nle_anno/VQA-X/vqaX_train.json | nle_anno/VQA-X/vqaX_val.json
    save_path: nle_anno/VQA-X/obj_train_labels.json | nle_anno/VQA-X/obj_val_labels.json
    """
    anno = json.load(open(args.anno_path, "r"))
    ids_list = list(anno.keys())

    labels = {}
    for i in tqdm(range(len(anno))):
        question_id = ids_list[i]
        sample = anno[question_id]
        img_name = sample["image_name"]

        img_path = os.path.join(args.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        inputs = inputs.to(args.gpu_id)
        outputs = model(**inputs)

        # keep only predictions with 0.7+ confidence
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        label = get_detected_objs(probas[keep], id2label)

        labels[img_name] = label

    return labels

def captioning_obj_labeling(model, id2label, img_lst, args):
    """
    image_dir: nocaps/image
    anno_path: 
    save_path: 
    """

    labels = {}
    for img_id in tqdm(img_lst):

        img_name = img_id + ".jpg"

        img_path = os.path.join(args.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        inputs = inputs.to(args.gpu_id)
        outputs = model(**inputs)

        # keep only predictions with 0.7+ confidence
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        label = get_detected_objs(probas[keep], id2label)

        labels[img_name] = label

    return labels



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="object detection")
    parser.add_argument("--dataset_name", type=str, required=True, help="vqax|nocaps|narratives|flickr30k|coco_caption")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--img_download", action="store_true", help= "just for nocaps image dataset")
    
    args = parser.parse_args()

    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    model = model.to(args.gpu_id)
    id2label = model.config.id2label

    # VQA-X NLE dataset
    if args.dataset_name=="vqax":
        if args.anno_path is None:
            raise ValueError("VQA-X needs annotation path for image_id")
        labels = VQA_X_obj_labeling(model, id2label, args)

    # nocaps captioning dataset
    elif args.dataset_name=="nocaps":
        # image downloading form URL
        id_file = {}
        ids_list = []

        anno = json.load(open(args.anno_path, "r"))
        for img in tqdm(anno["images"]):
            # URL -> *.jpg
            if args.img_download:
                file_name = os.path.join(args.image_dir, img["file_name"])
                url = img["coco_url"]
                try:
                    urllib.request.urlretrieve(url, file_name)
                except:
                    print(img)
            img_id = img["open_images_id"]
            id_file[img["id"]] = img_id
            ids_list.append(img_id)

        labels = captioning_obj_labeling(model, id2label, ids_list, args)

        for ann in tqdm(anno["annotations"]):
            ann_id = ann["image_id"]
            ann["image_id"] = id_file[ann_id]
        
        # Modify image id in annotations
        new_annotation_path = args.anno_path.split(".")[0] + "_caption.json"
        with open(args.anno_path, "w") as fout:
            json.dump(anno, fout, indent=2)

    # Localized narratives captioning dataset
    elif args.dataset_name=="narratives":
        annotation_dict = {}
        with jsonlines.open(args.anno_path) as f:
            for line in tqdm(f.iter()):
                img_id = line["image_id"]
                caption = line["caption"]
                annotation_dict[img_id] = caption
        ids_list = list(annotation_dict.keys())

        labels = captioning_obj_labeling(model, id2label, ids_list, args)
        
        # annotation datset updating
        new_annotation_path = args.anno_path.split(".")[0] + "_caption.json"
        with open(new_annotation_path, "w") as fout:
            json.dump(annotation_dict, fout, indent=2)
            
            
    # Flickr30k captioning dataset
    elif args.dataset_name == "flickr30k":
    
        annotation_dict = defaultdict(list)
        with open(args.anno_path, "r", ) as f:
            annotation = f.readlines()
        for anno in annotation:
            img_id, caption = re.split(r"#\d\t",anno)
            annotation_dict[img_id[:-4]].append(caption[:-1])
        ids_list = list(annotation_dict.keys())

        labels = captioning_obj_labeling(model, id2label, ids_list, args)
        
        # *.token -> *.json
        # {img_id : captioning list}
        new_annotation_path = args.anno_path.split(".")[0] + "_caption.json"
        with open(new_annotation_path, "w") as fout:
            json.dump(annotation_dict, fout, indent=2)
            
    # COCO captioning dataset
    elif args.dataset_name=="coco_caption":
        
        anno = json.load(open(args.anno_path, "r"))
        annotation_dict = defaultdict(list)
        for annotation in anno["annotations"]:
            # 12345 -> COCO_val2014_000000012345
            image_name = "COCO_val2014_" + (12 - len(str(annotation["image_id"]))) * '0' + str(annotation["image_id"])
            annotation_dict[image_name].append(annotation["caption"])
            
        ids_list = list(annotation_dict.keys())
        labels = captioning_obj_labeling(model, id2label, ids_list, args)
        
        # annotation datset updating
        new_annotation_path = args.anno_path.split(".")[0]+"_caption.json"
        with open(new_annotation_path, "w") as fout:
            json.dump(annotation_dict, fout, indent=2)            
    else:
        raise NotImplementedError
    
    with open(args.save_path, "w") as fout:
        json.dump(labels, fout, indent=2)