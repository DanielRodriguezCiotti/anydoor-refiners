import gradio as gr
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from refiners.fluxion.utils import manual_seed, no_grad
from anydoor_refiners.preprocessing import preprocess_images
from anydoor_refiners.postprocessing import post_processing
from anydoor_refiners.model import AnyDoor
import logging

# Initialize the logging and model setup
logging.basicConfig(level=logging.INFO)
logging.info("Setting up the model...")

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# model = AnyDoor(device=device, dtype=dtype)
# model.unet.load_from_safetensors("ckpt/refiners/unet.safetensors")
# model.control_model.load_from_safetensors("ckpt/refiners/controlnet.safetensors")
# model.object_encoder.load_from_safetensors("ckpt/refiners/dinov2_encoder.safetensors")
# model.lda.load_from_safetensors("ckpt/refiners/lda_new.safetensors")

def run_inference(object_image:np.ndarray, background_image:np.ndarray, background_mask:np.ndarray, seed=42, uncod_scale=5.0, num_inference_steps=50):
    # Load and preprocess images
    # Print types and shapes of input images
    logging.info(f"Object Image: {type(object_image)},{object_image.shape}")
    logging.info(f"Background Image: {type(background_image)}, {background_image.shape}")
    logging.info(f"Background Mask: {type(background_mask)}, {background_mask.shape}")

    object_mask = (object_image[:, :, -1] > 128).astype(np.uint8)
    object_image = object_image[:, :, :-1]
    object_image = cv2.cvtColor(object_image.copy(), cv2.COLOR_BGR2RGB)
    
    # background_image = cv2.imread(background_image_path).astype(np.uint8)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    
    # background_mask = cv2.imread(background_mask_path)[:, :, 0] > 128
    background_mask = background_mask.astype(np.uint8)
    
    preprocessed_images = preprocess_images(object_image, object_mask, background_image.copy(), background_mask)

    # Run inference
    control_tensor = torch.from_numpy(preprocessed_images['collage'].copy()).to(device=device, dtype=dtype).unsqueeze(0).permute(0, 3, 1, 2)
    object_tensor = torch.from_numpy(preprocessed_images['object'].copy()).to(device=device, dtype=dtype).unsqueeze(0).permute(0, 3, 1, 2)

    # with no_grad():  
    #     manual_seed(seed)
    #     object_embedding = model.object_encoder.forward(object_tensor)
    #     negative_object_embedding = model.object_encoder.forward(torch.zeros((1, 3, 224, 224), device=device, dtype=dtype))
    #     x = model.init_latents((512, 512))

    #     for step in tqdm(model.steps if num_inference_steps == model.steps else range(num_inference_steps)):
    #         x = model.forward(
    #             x,
    #             step=step,
    #             control_background_image=control_tensor,
    #             object_embedding=object_embedding,
    #             negative_object_embedding=negative_object_embedding,
    #             condition_scale=uncod_scale
    #         )
    #     predicted_image = model.lda.latents_to_image(x)

    # Post-processing
    generated_image = post_processing(
        np.zeros_like(background_image), 
        background_image,
        preprocessed_images["sizes"].tolist(),
        preprocessed_images["background_box"].tolist()
    )

    # Convert to RGB for display
    generated_image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
    collage = cv2.cvtColor((preprocessed_images["collage"][:,:,:-1]+1)/2, cv2.COLOR_RGB2BGR)
    return generated_image_rgb, collage


objects_list=[os.path.join('examples/objects',file) for file in os.listdir('examples/objects') if '.jpg' in file or '.png' in file or '.jpeg' in file ]
objects_list.sort()

backgrounds_list=[os.path.join('examples/backgrounds',file) for file in os.listdir('examples/backgrounds') if '.jpg' in file or '.png' in file or '.jpeg' in file ]
backgrounds_list.sort()

background_masks_list=[os.path.join('examples/background_masks',file) for file in os.listdir('examples/background_masks') if '.jpg' in file or '.png' in file or '.jpeg' in file ]
background_masks_list.sort()




# Set up Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with AnyDoor")
    with gr.Row():
        object_image_input = gr.Image(sources="upload", label="Upload Object Image (with Alpha Channel)", image_mode="RGBA")
        background_image_input = gr.Image(sources="upload", label="Upload Background Image")
        background_mask_input = gr.Image(sources="upload", label="Upload Background Mask Image")
        conditionning_image = gr.Image(label="Conditionning Image",image_mode="RGB", type="numpy")

    with gr.Accordion("Advanced Option", open=True):
        seed_input = gr.Number(value=42, label="Seed")
        scale_input = gr.Number(value=5.0, label="Uncod Scale")
        steps_input = gr.Number(value=50, label="Inference Steps")

    generate_button = gr.Button("Generate Image")

    output_image = gr.Image(label="Generated Image")

    generate_button.click(
        run_inference,
        inputs=[object_image_input, background_image_input, background_mask_input, seed_input, scale_input, steps_input],
        outputs=[output_image,conditionning_image]
    )

    with gr.Row():
        with gr.Column():
            gr.Examples(objects_list, inputs=[object_image_input],label="Examples - Object",examples_per_page=6)
        with gr.Column():
            gr.Examples(backgrounds_list, inputs=[background_image_input],label="Examples - Background",examples_per_page=6)
        with gr.Column():
            gr.Examples(background_masks_list, inputs=[background_mask_input],label="Examples - Background Mask",examples_per_page=6)

# Launch the Gradio interface
demo.launch()