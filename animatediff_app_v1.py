
import os
import datetime
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from chatgpt import ChatGPTBot
from video_processor import MoiveEditor


#sample_idx     = 0
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "./")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir                = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample         = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list   = []
        self.motion_module_list      = []
        self.personalized_model_list = []
        
        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # chatgpt config
        self.bot = ChatGPTBot()
        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.lora_model_state_dict = {}
        
        self.inference_config      = OmegaConf.load("configs/inference/inference-v2.yaml")
        stable_diffusion_dropdown = '/home/zhangyan/huyong/AnimateDiffV0/models/StableDiffusion/stable-diffusion-v1-5/'
        motion_module_dropdown = '2mmsd.ckpt'
        base_model_dropdown = 'toonyou_beta3.safetensors'
        #base_model_dropdown = 'realisticVisionV51_v51VAE.safetensors'

        self.update_stable_diffusion(stable_diffusion_dropdown)
        self.update_motion_module(motion_module_dropdown)
        self.update_base_model(base_model_dropdown)
        self.video_editor = MoiveEditor()

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
                    
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()

    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_state_dict = {}
        if lora_model_dropdown == "none": pass
        else:
            with safe_open(lora_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.lora_model_state_dict[key] = f.get_tensor(key)
        return gr.Dropdown.update()
    
    def make_story(
        self,
        keywords
    ):    
        stories = self.bot.make_story(keywords)
        #stories = self.bot.extract_prompts(chatgpt_resp)
        return stories
    
    def make_prompt(
        self,
        prompt_textbox
    ):    
        prompts = self.bot.make_prompt(prompt_textbox)
        #prompts = self.bot.extract_prompts(chatgpt_resp)
        return prompts
    
    def make_dir(self, seed):
        # Get the current date and time
        current_datetime = datetime.now()

        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        # Combine the base directory path and formatted datetime to create the new directory path
        new_directory = os.path.join(self.savedir, formatted_datetime)
        # Check if the directory already exists; if not, create it
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            print(f"Created directory: {new_directory}")
        else:
            print(f"Directory already exists: {new_directory}")
        return new_directory

    def animate_one(
        self,
        prompt_textbox,
        seed_textbox,
        savedir_sample,
        sample_idx,
        width_slider,
        height_slider
    ):    
        lora_alpha_slider = '0.8'
        negative_prompt_textbox = 'badhandv4,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg, worst quality, low quality, deformed, distorted, disfigured, bad eyes, wrong lips, weird mouth, bad teeth, mutated hands and fingers, bad anatomy, wrong anatomy, amputation, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-negd'
        sampler_dropdown = 'Euler'
        sample_step_slider = 25
        length_slider = 16
        cfg_scale_slider = 7.5
        if self.unet is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")
        
        if self.lora_model_state_dict != {}:
            pipeline = convert_lora(pipeline, self.lora_model_state_dict, alpha=lora_alpha_slider)

        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: torch.seed()
        seed = torch.initial_seed()
        
        sample = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = sample_step_slider,
            guidance_scale      = cfg_scale_slider,
            width               = width_slider,
            height              = height_slider,
            video_length        = length_slider,
        ).videos

        save_sample_path = os.path.join(savedir_sample, f"{seed_textbox}_{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)
        
    
        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(savedir_sample, f"{seed_textbox}_{sample_idx}_logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return save_sample_path
    
    def animate(
        self,
        prompt_textbox,
        story_textbox,
        width_slider,
        height_slider
    ):    
        prompts = []
        stories = []
        for line in prompt_textbox.split('\n'):
            if line.strip() != "":
                prompts.append(line.strip())
        for line in story_textbox.split('\n'):
            if line.strip() != "":
                stories.append(line.strip())
        min_length = min(len(prompts), len(stories))
        prompts = prompts[:min_length]
        stories = stories[:min_length]
        sample_idx = 0
        output_videos = []
        for i in range(2):
            seed_textbox = random.randint(1, 1e8)
            savedir_sample =  self.make_dir(seed_textbox)
            videos = []
            for prompt in prompts:
                print("Generate Prompt[seed={}]: {}".format(seed_textbox, prompt))
                video = self.animate_one(prompt,
                                seed_textbox,
                                savedir_sample,
                                sample_idx,
                                width_slider,
                                height_slider
                                )
                # video = "./samples/Gradio-2023-09-17T20-48-30/2023-09-17_20-49-32/3410905_{}.mp4".format(sample_idx)
                sample_idx += 1
                videos.append(video)
            outfile_path = savedir_sample + f"{seed_textbox}_final.mp4"
            finalvideo = self.video_editor.process(videos, stories, outfile_path)
            output_videos.append(outfile_path)
            print('Finish {} {} {}'.format(seed_textbox, outfile_path, outfile_path))
        return [gr.Video.update(value=path) for path in output_videos]
            

controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        with gr.Column(variant="panel2"):
            # gr.Markdown(
            #     """
            #     1. 输入故事大纲（留白表示随机生成）， 和商品
            #     2. 点击生成后等待
            #     """
            # )
            with gr.Row(variant="panel1").style():
                prompt_textbox = gr.Textbox(label="商品简介", lines=2)
                with gr.Column():
                    with gr.Row():                  
                        width_slider     = gr.Slider(label="Width",            value=512, minimum=256, maximum=1024, step=64)
                        height_slider    = gr.Slider(label="Height",           value=512, minimum=256, maximum=1024, step=64)
                        length_slider    = gr.Slider(label="Animation length", value=16,  minimum=8,   maximum=24,   step=1)
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
            with gr.Row(variant="panel2").style():
                generate_story_button = gr.Button(value="生成剧本", variant='primary')
                generate_prompt_button = gr.Button(value="生成提示词", variant='primary')
            with gr.Row(variant="panel3").style():
                story = gr.Textbox(label="剧本", lines=5)
                sd_prompts = gr.Textbox(label="提示词", lines=5)
            generate_story_button.click(
                    fn=controller.make_story,
                    inputs=[
                        prompt_textbox
                    ],
                    outputs=[story])
            generate_prompt_button.click(
                    fn=controller.make_prompt,
                    inputs=[
                        story
                    ],
                    outputs=[sd_prompts])
                

        with gr.Column(variant="panel2"):
            with gr.Row(variant="panel3").style():
                generate_video_button = gr.Button(value="生成视频", variant='primary')
            with gr.Row().style(equal_height=True):
                result_video1 = gr.Video(label="Generated Animation 1", interactive=False)
                result_video2 = gr.Video(label="Generated Animation 2", interactive=False)
            # with gr.Row().style(equal_height=True):
            #     result_video3 = gr.Video(label="Generated Animation 3", interactive=False)
            #     result_video4 = gr.Video(label="Generated Animation 4", interactive=False)
            generate_video_button.click(
                    fn=controller.animate,
                    inputs=[
                        sd_prompts,
                        story,
                        width_slider,
                        height_slider
                    ],
                    outputs=[result_video1, result_video2]  
                                    )
                

            
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue().launch(share=True, server_port=1798)
