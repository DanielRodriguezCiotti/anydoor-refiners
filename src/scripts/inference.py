import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from refiners.fluxion.utils import manual_seed, no_grad, load_from_safetensors
from anydoor_refiners.lora import build_lora, set_lora_weights
from anydoor_refiners.postprocessing import post_processing
from anydoor_refiners.model import AnyDoor
from training.data.batch import AnyDoorBatch, collate_fn

from loguru import logger

from training.data.vitonhd import CustomDataLoader, VitonHDDataset

# Set up logging
torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
logger.info(f"Using device: {device}")
logger.info(f"Using dtype: {dtype}")
seed = 42
uncod_scale = 5.0
num_inference_steps = 50

def inference(
    output_folder_path:str,lora_ckpt: str | None = None, lora_rank: int = 16, lora_scale: float = 1.0
):
    
    logger.info("Loading dataset")
    dataset = VitonHDDataset("dataset/test/cloth", filtering_file="dataset/lora_test_images.txt", inference=True)
    dataloader = CustomDataLoader(dataset, batch_size=5, collate_fn=collate_fn,shuffle=True)

    logger.info("Setting up the model")
    model = AnyDoor(device=device, dtype=dtype)
    logger.info("Loading weights ...")
    model.unet.load_from_safetensors("ckpt/refiners/unet.safetensors")
    model.control_model.load_from_safetensors("ckpt/refiners/controlnet.safetensors")
    model.object_encoder.load_from_safetensors(
        "ckpt/refiners/dinov2_encoder.safetensors"
    )
    model.lda.load_from_safetensors("ckpt/refiners/lda_new.safetensors")
    if lora_ckpt:
        logger.info(f"Loading LORA weights from {lora_ckpt}")
        build_lora(model, lora_rank, lora_scale)
        lora_weights = load_from_safetensors(lora_ckpt)
        set_lora_weights(model.unet, lora_weights)
    

    if num_inference_steps != model.steps:
        model.set_inference_steps(num_inference_steps, first_step=0)

    predicted_images = {}
    logger.info("Starting inference ...")
    for batch in tqdm(dataloader):
        assert batch is not None
        assert type(batch) is AnyDoorBatch
        assert batch.background_image is not None
        object = batch.object.to(device, dtype)
        collage = batch.collage.to(device, dtype)
        batch_size = object.shape[0]
        with no_grad():
            manual_seed(seed)
            object_embedding = model.object_encoder.forward(object)
            negative_object_embedding = model.object_encoder.forward(
                torch.zeros(
                    (batch_size, 3, 224, 224), device=device, dtype=dtype
                )
            )
            x = model.sample_noise(
                (batch_size, 4, 512 // 8, 512 // 8),
                device=device,
                dtype=dtype,
            )

            for step in model.steps:   
                x = model.forward(
                    x,
                    step=step,
                    control_background_image=collage,
                    object_embedding=object_embedding,
                    negative_object_embedding=negative_object_embedding,
                    condition_scale=uncod_scale,
                )

            background_images = batch.background_image.numpy()
            for j in range(batch_size):
                filename = batch.filename[j]
                predicted_images[filename] = {
                    "predicted_image": model.lda.latents_to_image(x[j].unsqueeze(0)),
                    "ground_truth": background_images[j],
                    "sizes": batch.sizes.tolist()[j],
                    "background_box": batch.background_box.tolist()[j],
                }
    
    logger.info("Saving images ...")
    for filename,predicted_image in tqdm(predicted_images.items()):
        generated_image = Image.fromarray(
            post_processing(
                np.array(predicted_image["predicted_image"]),
                predicted_image["ground_truth"],
                predicted_image["sizes"],
                predicted_image["background_box"],
            )
        )
        image_output_path = os.path.join(output_folder_path, filename)
        generated_image.save(image_output_path)
    logger.info("Inference finished")
    

if __name__ == "__main__":
    lora_ckpt = None
    rank = 16
    scale = 1.0
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_ckpt", type=str, default=lora_ckpt, required=False)
    parser.add_argument("--rank", type=int, default=rank, required= False)
    parser.add_argument("--scale", type=float, default=scale, required=False)
    args = parser.parse_args()
    lora_ckpt = args.lora_ckpt
    rank = args.rank
    scale = args.scale
    if lora_ckpt:
        output_folder_path = "dataset/generated/" + os.path.basename(lora_ckpt).replace(".safetensors", "")
    else:
        output_folder_path = "dataset/generated/anydoor"
    os.makedirs(output_folder_path, exist_ok=True)
    inference(output_folder_path, lora_ckpt, rank, scale)