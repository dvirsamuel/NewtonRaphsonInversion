import torch
from matplotlib import pyplot as plt

from src.config import RunConfig
import PIL
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

def inversion_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs


def inference_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs

def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(PIL.Image.open(im_path)).resize((512, 512))

class ImageEditorDemo:
    def __init__(self, pipe_inversion, pipe_inference, input_image, description_prompt, cfg):
        self.pipe_inversion = pipe_inversion
        self.pipe_inference = pipe_inference
        self.original_image = load_im_into_format_from_path(input_image).convert("RGB")
        self.load_image = True
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = (512,512)
        VQAE_SCALE = 8
        latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
        noise = [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=g_cpu) for i
                 in range(cfg.num_inversion_steps)]
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
        pipe_inversion.set_progress_bar_config(disable=True)
        pipe_inference.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.pipe_inversion.cfg = cfg
        self.pipe_inference.cfg = cfg
        self.inv_hp = [2, 0.1, 0.2]
        self.edit_cfg = 1.2

        self.pipe_inference.to("cuda")
        self.pipe_inversion.to("cuda")

        self.last_latent = self.invert(self.original_image, description_prompt)
        self.original_latent = self.last_latent

    def invert(self, init_image, base_prompt):
        res = self.pipe_inversion(prompt=base_prompt,
                             num_inversion_steps=self.cfg.num_inversion_steps,
                             num_inference_steps=self.cfg.num_inference_steps,
                             image=init_image,
                             guidance_scale=self.cfg.guidance_scale,
                             callback_on_step_end=inversion_callback,
                             strength=self.cfg.inversion_max_step,
                             denoising_start=1.0 - self.cfg.inversion_max_step,
                             inv_hp=self.inv_hp)[0][0]
        return res

    def edit(self, target_prompt):
        image = self.pipe_inference(prompt=target_prompt,
                            num_inference_steps=self.cfg.num_inference_steps,
                            negative_prompt="",
                            callback_on_step_end=inference_callback,
                            image=self.last_latent,
                            strength=self.cfg.inversion_max_step,
                            denoising_start=1.0 - self.cfg.inversion_max_step,
                            guidance_scale=self.edit_cfg).images[0]
        return image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (512, 512)
    scheduler_class = MyEulerAncestralDiscreteScheduler
    pipe_inversion = SDXLDDIMPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True,
                                                      safety_checker=None, cache_dir="/inputs/huggingface_cache").to(
        device)
    pipe_inference = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True,
                                                                safety_checker=None,
                                                                cache_dir="/inputs/huggingface_cache").to(device)
    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)
    pipe_inversion.scheduler_inference = scheduler_class.from_config(pipe_inference.scheduler.config)

    config = RunConfig(num_inference_steps=4,
                       num_inversion_steps=4,
                       guidance_scale=0.0,
                       inversion_max_step=0.6)

    input_image = "example_images/lion.jpeg"
    description_prompt = 'a lion is sitting in the grass at sunset'
    editor = ImageEditorDemo(pipe_inversion, pipe_inference, input_image, description_prompt, config)

    editing_prompt = "a raccoon is sitting in the grass at sunset"
    plt.imshow(editor.edit(editing_prompt))
    plt.show()