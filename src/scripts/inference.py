import torch
import cv2
import numpy as np

from tqdm import tqdm
from refiners.fluxion.utils import manual_seed, no_grad
from anydoor_refiners.preprocessing import preprocess_images
from anydoor_refiners.postprocessing import post_processing
from anydoor_refiners.model import AnyDoor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

logging.info("Setting up the model")

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

logging.info(f"Using device: {device}")
logging.info(f"Using dtype: {dtype}")

logging.info("Loading weights ...")
model = AnyDoor(device=device,dtype=dtype)
model.unet.load_from_safetensors("ckpt/refiners/unet.safetensors")
model.control_model.load_from_safetensors("ckpt/refiners/controlnet.safetensors")
model.object_encoder.load_from_safetensors("ckpt/refiners/dinov2_encoder.safetensors")
model.lda.load_from_safetensors("ckpt/refiners/lda_new.safetensors")

seed = 42
uncod_scale = 5.0
num_inference_steps = 50
if num_inference_steps!= model.steps:
    model.set_inference_steps(num_inference_steps, first_step=0)

logging.info("Preprocessing images ...")
background_image_path = 'examples/background.png'
background_mask_path = 'examples/background_mask.png'
object_image_path = 'examples/object.png'

object_image = cv2.imread(object_image_path, cv2.IMREAD_UNCHANGED)
object_mask = (object_image[:,:,-1] > 128).astype(np.uint8)
object_image = object_image[:,:,:-1]
object_image = cv2.cvtColor(object_image.copy(), cv2.COLOR_BGR2RGB)
background_image = cv2.imread(background_image_path).astype(np.uint8)
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
background_mask = cv2.imread(background_mask_path)[:,:,0] > 128
background_mask = background_mask.astype(np.uint8)

preprocessed_images = preprocess_images(object_image, object_mask, background_image.copy(), background_mask)


logging.info("Generating image ...")

control_tensor = torch.from_numpy(preprocessed_images['collage'].copy()).to(device=device,dtype=dtype).unsqueeze(0).permute(0,3,1,2)
object_tensor = torch.from_numpy(preprocessed_images['object'].copy()).to(device=device ,dtype=dtype).unsqueeze(0).permute(0,3,1,2)

with no_grad():  
    manual_seed(seed)
    object_embedding = model.object_encoder.forward(object_tensor)
    negative_object_embedding = model.object_encoder.forward(torch.zeros((1, 3, 224, 224),device=device,dtype=dtype))
    x = model.init_latents((512, 512))

    for step in tqdm(model.steps):
        x = model.forward(
            x,
            step=step,
            control_background_image= control_tensor,
            object_embedding= object_embedding,
            negative_object_embedding= negative_object_embedding,
            condition_scale= 5.0
        )
    predicted_image = model.lda.latents_to_image(x)

generated_image = post_processing(np.array(predicted_image),background_image,preprocessed_images["sizes"].tolist(),preprocessed_images["background_box"].tolist())

logging.info("Saving image ...")

cv2.imwrite("generated_image.png", cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
