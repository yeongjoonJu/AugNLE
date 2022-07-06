import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from self_training.models.modules import ImageEncoder
import torchvision.transforms as transforms
from PIL import Image
import torch


def caching(args):
    img_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_encoder = ImageEncoder(args.gpu_id).to(args.gpu_id)
    
    img_list = glob(args.image_dir + "/*.jpg")
    img_list.extend(glob(args.image_dir+"/*.png"))
    num_imgs = len(img_list)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for b in tqdm(range(0, num_imgs, args.batch_size)):
        # Construct batch image
        batch_image = []
        n_items = args.batch_size if num_imgs-(b+args.batch_size) >= 0 else num_imgs-b
        for i in range(n_items):
            img_path = img_list[b+i]
            img = Image.open(img_path).convert("RGB")
            img = img_transform(img)
            batch_image.append(img.unsqueeze(0))
        batch_image = torch.cat(batch_image, dim=0).to(args.gpu_id)

        features = img_encoder(batch_image).cpu().numpy()

        for i in range(n_items):
            img_name = os.path.basename(img_list[b+i])
            np.save(os.path.join(args.save_dir, img_name), features[i])
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Image encoding")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    
    caching(args)