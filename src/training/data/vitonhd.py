import cv2
import numpy as np
import os
import torch
from PIL import Image 
from .base import BaseDataset
class VitonHDDataset(BaseDataset):
    def __init__(self, image_dir, filtering_file = None, inference = False):
        self.image_root = image_dir
        self.data = os.listdir(self.image_root)
        if filtering_file is not None:
            self.filter_data(filtering_file)
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.inference = inference
    
    def filter_data(self, filtering_file):
        with open(filtering_file, "r") as f:
            lines = f.readlines()
        labelled_values = {line.split(",")[0]: line.split(",")[1].strip() for line in lines}
        images_to_keep = [ image for image,label in labelled_values.items() if label == "True"]
        self.data = [image for image in self.data if image in images_to_keep]

    def __len__(self):
        return len(self.data)

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag
            
    def get_sample(self, idx):

        ref_image_path = os.path.join(self.image_root, self.data[idx])
        tar_image_path = ref_image_path.replace('/cloth/', '/image/')
        ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
        tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 5

        try:
            item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        except Exception as _:
            print(f"Error in processing with {ref_image_path}")
            return None
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps


        batch = dict(
            object = torch.from_numpy(item_with_collage['ref']).permute(2,0,1),
            background = torch.from_numpy(item_with_collage['jpg']).permute(2,0,1),
            collage=torch.from_numpy(item_with_collage['hint']).permute(2,0,1),
            background_box=torch.from_numpy(item_with_collage['tar_box_yyxx_crop']),
            sizes=torch.from_numpy(item_with_collage['extra_sizes']),
            time_steps=torch.from_numpy(sampled_time_steps),
        )

        if self.inference:
            batch["background_image"] = torch.from_numpy(tar_image)
            return batch
        else:
            return batch



if __name__ == "__main__":
    dataset = VitonHDDataset("dataset/train/cloth", filtering_file="dataset/lora_training_images.txt")
    print(len(dataset))