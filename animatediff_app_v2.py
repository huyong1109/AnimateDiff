
import os
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


sample_idx     = 0
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
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
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
        self.video_editor = MoiveEditor()
        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.lora_model_state_dict = {}
        
        self.inference_config      = OmegaConf.load("configs/inference/inference-v2.yaml")

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
        results = prompts.split('\n')
        if len(results) < 6:
            for i in range(6 - len(results)):
                results.append("")
        return results
    
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
    
    def animate(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox,
        sample_idx
    ):    
        if self.unet is None:
            raise gr.Error(f"Please select a pretrained model path.")
        if motion_module_dropdown == "": 
            raise gr.Error(f"Please select a motion module.")
        if base_model_dropdown == "":
            raise gr.Error(f"Please select a base DreamBooth model.")

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
        # sample = gr.Video('/home/zhangyan/huyong/AnimateDiffV0/samples/Gradio-2023-10-16T06-44-02/sample/-1_.mp4')

        save_sample_path = os.path.join(self.savedir_sample, f"{seed_textbox}_{sample_idx}.mp4")
        print('save video {}'.format(save_sample_path))
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
            "seed": seed,
            "path": save_sample_path
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return gr.Video.update(value=save_sample_path)
    
    def animate_all(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_alpha_slider,
        sd_prompts,
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox,
        sample_index
    ):
        result_videos = []
        sample_index = 0
        print("negprompt : {}\n".format(negative_prompt_textbox))
        for x in sd_prompts.split('\n'):
            print("prompt : {}".format(x))
            result_video = self.animate(
                stable_diffusion_dropdown,
                motion_module_dropdown,
                base_model_dropdown,
                lora_alpha_slider,
                x,
                negative_prompt_textbox, 
                sampler_dropdown, 
                sample_step_slider, 
                width_slider, 
                length_slider, 
                height_slider, 
                cfg_scale_slider, 
                seed_textbox,
                sample_index
            )
            result_videos.append(result_video)  
            sample_index += 1
        return result_videos
    def regenerate(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_alpha_slider,
        prompt,
        negative_prompt_textbox, 
        sampler_dropdown,
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox
    ):
            
            result_video = self.animate(
                self,
                stable_diffusion_dropdown,
                motion_module_dropdown,
                base_model_dropdown,
                lora_alpha_slider,
                prompt,
                negative_prompt_textbox, 
                sampler_dropdown, 
                sample_step_slider, 
                width_slider, 
                length_slider, 
                height_slider, 
                cfg_scale_slider, 
                seed_textbox
            )

    def merge_videos(self,
        prompt,
        seed_textbox,
        image):
        outfile_path = os.path.join(self.savedir_sample, f"{prompt}_{seed_textbox}_merge.mp4")
        finalvideo = self.video_editor.merge(self.savedir_sample, prompt, seed_textbox, image)
        return gr.Video.update(value=outfile_path)

controller = AnimateController()


def ui():
    max_textboxes = 10
    with gr.Blocks(css=css) as demo:
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 模型配置 (第一步先加载Pretrained模型路径).
                """
            )
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained模型路径",
                    choices=controller.stable_diffusion_list,
                    interactive=True,
                )
                stable_diffusion_dropdown.change(fn=controller.update_stable_diffusion, inputs=[stable_diffusion_dropdown], outputs=[stable_diffusion_dropdown])
                
                stable_diffusion_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_stable_diffusion():
                    controller.refresh_stable_diffusion()
                    return gr.Dropdown.update(choices=controller.stable_diffusion_list)
                stable_diffusion_refresh_button.click(fn=update_stable_diffusion, inputs=[], outputs=[stable_diffusion_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Motion模型(必选)",
                    choices=controller.motion_module_list,
                    interactive=True,
                )
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])
                
                motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.Dropdown.update(choices=controller.motion_module_list)
                motion_module_refresh_button.click(fn=update_motion_module, inputs=[], outputs=[motion_module_dropdown])
                
                base_model_dropdown = gr.Dropdown(
                    label="基线Dreambooth模型 (必选)",
                    choices=controller.personalized_model_list,
                    interactive=True,
                )
                base_model_dropdown.change(fn=controller.update_base_model, inputs=[base_model_dropdown], outputs=[base_model_dropdown])
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                lora_model_dropdown.change(fn=controller.update_lora_model, inputs=[lora_model_dropdown], outputs=[lora_model_dropdown])
                
                lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateDiff.
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt", lines=2)
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2)
                
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
                    # max_textboxes_button = gr.Slider(label="Frame Num", value=max_textboxes, minimum=1, maximum=10, step=1)
                    
                with gr.Column():
                    width_slider     = gr.Slider(label="Width",            value=512, minimum=256, maximum=1024, step=64)
                    height_slider    = gr.Slider(label="Height",           value=512, minimum=256, maximum=1024, step=64)
                    length_slider    = gr.Slider(label="Animation length", value=16,  minimum=8,   maximum=24,   step=1)
                    cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.5, minimum=0,   maximum=20)
            stories = []
            sd_prompts = []
            max_textboxes = 6
            with gr.Row().style(equal_height=False):
                    generate_story_button = gr.Button(value="生成剧本", variant='primary')
                    
                    generate_prompt_button = gr.Button(value="生成提示词", variant='primary')
            with gr.Row(variant="panel3").style():
                story = gr.Textbox(label="剧本", lines=max_textboxes)
                sd_prompts = []
                with gr.Column():
                    for i in range(max_textboxes):
                        sd_prompt = gr.Textbox(label="提示词{}".format(i), lines=1)
                        sd_prompts.append(sd_prompt)
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
                    outputs=sd_prompts)
            with gr.Row(variant="panel3").style():
                generate_button = gr.Button(value="一键生成", variant='primary')
                merge_button = gr.Button(value="合成视频", variant='primary')
            
        frame_generate_buttons = []
        frame_result_videos = []
        indexs = []
        # with gr.Row(variant="panel2"):
        #     generate_button = gr.Button(value="ReGen")
        #     for i in range(6):
        #         with gr.Column():
        #             generate_button = gr.Button(value="Video{}".format(i))
        #             #result_video = gr.Video(label="Animation {}".format(i), interactive=False)
        #             frame_generate_buttons.append(generate_button)
        #             #frame_result_videos.append(result_video)
        with gr.Row(variant="panel3"):
            for i in range(6):
                with gr.Column():
                    #
                    result_video = gr.Video(label="Animation {}".format(i), interactive=False)
                    index = gr.Text(value='{}'.format(i), visible=False)
                    indexs.append(index)
                    frame_result_videos.append(result_video)
                    regenerate_button = gr.Button(value="ReGen")
                    regenerate_button.click(
                                    fn=controller.animate,
                                    inputs=[
                                        stable_diffusion_dropdown,
                                        motion_module_dropdown,
                                        base_model_dropdown,
                                        lora_alpha_slider, 
                                        sd_prompts[i],
                                        negative_prompt_textbox, 
                                        sampler_dropdown, 
                                        sample_step_slider, 
                                        width_slider, 
                                        length_slider, 
                                        height_slider, 
                                        cfg_scale_slider,
                                        seed_textbox,
                                        indexs[i]
                                    ],
                                    outputs=[frame_result_videos[i]]
                                )            
        
        generate_button.click(
                fn=controller.animate_all,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_alpha_slider, 
                    sd_prompts[0],
                    sd_prompts[1],
                    sd_prompts[2],
                    sd_prompts[3],
                    sd_prompts[4],
                    sd_prompts[5],
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider,
                    seed_textbox,
                ],
                outputs=frame_result_videos
            )
        with gr.Row(variant="panel3"):
            input_image = gr.Image(shape=(512, 512))
            result_video = gr.Video(label="Generated Animation", interactive=False)
        merge_button.click(
            fn=controller.merge_videos,
            inputs=[
                prompt_textbox,
                seed_textbox,
                input_image
            ],
            outputs = result_video
        )
            
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue().launch(share=False, server_port=1798)
