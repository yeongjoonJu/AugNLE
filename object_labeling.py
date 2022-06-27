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

def load_detection_model():
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    id2label = model.config.id2label

    return feature_extractor, model, id2label


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


def get_object_labels(captions, args):
    feature_extractor, model, id2label = load_detection_model()
    model = model.to(args.gpu_id)

    labels = {}
    for img_name, caps in tqdm(captions.items()):
        if img_name[-4:]!=".jpg":
            img_name = img_name + ".jpg"

        img_path = os.path.join(args.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(args.gpu_id)
        outputs = model(**inputs)

        # keep only predictions with 0.7+ confidence
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        label = get_detected_objs(probas[keep], id2label)

        labels[img_name] = {"obj_label": label, "captions": caps}

    return labels


def vqa_x_obj_labeling(args):
    """
    image_dir: images/train2014 | images/val2014
    anno_path: nle_anno/VQA-X/vqaX_train.json | nle_anno/VQA-X/vqaX_val.json
    save_path: nle_anno/VQA-X/obj_train_labels.json | nle_anno/VQA-X/obj_val_labels.json
    """
    anno = json.load(open(args.anno_path, "r"))
    ids_list = list(anno.keys())

    img_list = []
    for i in range(len(anno)):
        q_id = ids_list[i]
        sample = anno[q_id]
        img_list.append(sample["image_name"])

    return get_object_labels(img_list, args)


def nocaps_obj_labeling(args):
    """
    image_dir: captioning_data/nocaps/image
    anno_path: captioning_data/nocaps/annotations/nocaps_val_4500_captions.json
    save_path: captioning_data/nocaps/obj_labels/nocaps.json
    """
    # image downloading form URL
    id_to_img_name = {}
    download = len(os.listdir(args.image_dir))==0
    anno = json.load(open(args.anno_path, "r"))
    for img in tqdm(anno["images"]):
        if download:
            filename = os.path.join(args.image_dir, img["file_name"])
            url = img["coco_url"]
            try:
                urllib.request.urlretrieve(url, filename)
            except:
                print("Fail to download", url)
        
        id_to_img_name[img["id"]] = img["file_name"]

    captions = {}
    for sample in anno["annotations"]:
        key = id_to_img_name[sample["image_id"]]
        if key in captions:
            captions[key].append(sample["caption"])
        else:
            captions[key] = [sample["caption"]]

    return get_object_labels(captions, args)


def narratives_obj_labeling(args):
    """
    image_dir: captioning_data/narratives/image/validation
    anno_path: captioning_data/narratives/annotations/open_images_validation_captions.jsonl
    save_path: captioning_data/narratives/obj_labels
    """
    annotation_dict = {}
    with jsonlines.open(args.anno_path) as f:
        for line in tqdm(f.iter()):
            img_id = line["image_id"]
            caption = line["caption"]
            annotation_dict[img_id] = caption
    ids_list = list(annotation_dict.keys())

    return get_object_labels(ids_list, args)


def flickr30k_obj_labeling(args):
    """
    image_dir: captioning_data/flickr30k/image
    anno_path: captioning_data/flickr30k/annotations/results_20130124.token
    save_path: captioning_data/flickr30k/obj_labels
    """
    captions = {}
    with open(args.anno_path, "r", ) as f:
        anno = f.readlines()
    for sample in anno:
        img_id, caption = re.split(r"#\d\t",sample)
        if img_id in captions:
            captions[img_id].append(caption[:-1])
        else:
            captions[img_id] = [caption[:-1]]

    return get_object_labels(captions, args)


def coco_obj_labeling(args):
    """
    image_dir: images/[train2014|val2014]
    anno_path: captioning_data/coco/annotations/captions_[train|val]2014.json
    save_path: captioning_data/coco/obj_labels
    """
    prepend = "COCO_val2014_" if "val" in args.anno_path else "COCO_train2014_"
    anno = json.load(open(args.anno_path, "r"))
    captions = {}
    for sample in anno["annotations"]:
        image_id = str(sample["image_id"])
        # 12345 -> COCO_val2014_000000012345
        image_name = prepend + (12 - len(image_id)) * '0' + image_id
        if image_name in captions:
            captions[image_name].append(sample["caption"])
        else:
            captions[image_name] = [sample["caption"]]
        
    return get_object_labels(captions, args)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="object detection")
    parser.add_argument("--dataset_name", type=str, required=True, help="vqax|nocaps|narratives|flickr30k|coco_caption")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True, help="Path to explanation label file")
    parser.add_argument("--gpu_id", type=int, default=1)
    
    args = parser.parse_args()

    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    model = model.to(args.gpu_id)
    id2label = model.config.id2label

    # VQA-X NLE dataset
    if args.dataset_name=="vqax":
        if args.anno_path is None:
            raise ValueError("VQA-X needs annotation path for image_id")
        labels = vqa_x_obj_labeling(args)

    # nocaps captioning dataset
    elif args.dataset_name=="nocaps":
        labels = nocaps_obj_labeling(args)

    # Localized narratives captioning dataset
    elif args.dataset_name=="narratives":
        labels = narratives_obj_labeling(args)            
            
    # Flickr30k captioning dataset
    elif args.dataset_name == "flickr30k":
        labels = flickr30k_obj_labeling(args)
            
    # COCO captioning dataset
    elif args.dataset_name=="coco":
        labels = coco_obj_labeling(args)
    else:
        raise NotImplementedError
    
    with open(args.save_path, "w") as fout:
        json.dump(labels, fout, indent=2)