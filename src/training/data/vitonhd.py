import cv2
import numpy as np
import os
import torch
from PIL import Image

from anydoor_refiners.preprocessing import preprocess_images 
from .base import BaseDataset
class VitonHDDataset(BaseDataset):
    def __init__(self, image_dir, filtering_file = None, inference = False):
        self.image_root = image_dir
        self.data = sorted(os.listdir(self.image_root))
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
        self.data = sorted(list(set(image for image in self.data if image in images_to_keep)))

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
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index out of range: {idx}. Dataset length: {len(self.data)}.")
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
            item_with_collage = preprocess_images(ref_image, ref_mask, tar_image, tar_mask)
        except Exception as _:
            print(f"Error in processing with {ref_image_path}")
            return None
        # sampled_time_steps = self.sample_timestep()
        # item_with_collage['time_steps'] = sampled_time_steps


        batch = dict(
            filename = self.data[idx],
            object = torch.from_numpy(item_with_collage['object']).permute(2,0,1),
            background = torch.from_numpy(item_with_collage['background']).permute(2,0,1),
            collage=torch.from_numpy(item_with_collage['collage']).permute(2,0,1),
            background_box=torch.from_numpy(item_with_collage['background_box']),
            sizes=torch.from_numpy(item_with_collage['sizes']),
            # time_steps=torch.from_numpy(sampled_time_steps),
        )

        if self.inference:
            batch["background_image"] = torch.from_numpy(tar_image)
            return batch
        else:
            return batch



class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.collate_fn = collate_fn
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0  # Tracks the current position in the dataset

    def __len__(self):
        # Total number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.current_index = 0  # Reset index for new iteration
        if self.shuffle:
            np.random.shuffle(self.indices)  # Reshuffle at the start of every epoch
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        # Determine the batch indices
        batch_indices = self.indices[self.current_index : self.current_index + self.batch_size]
        self.current_index += self.batch_size

        # Fetch the corresponding samples
        batch = [self.dataset.get_sample(idx) for idx in batch_indices]
        batch = [item for item in batch if item is not None]  # Filter out None values
        
        if len(batch) == 0:
            return None
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        return batch


if __name__ == "__main__":
    dataset = VitonHDDataset("dataset/test/cloth", filtering_file="dataset/lora_test_images.txt")
    print(len(dataset))