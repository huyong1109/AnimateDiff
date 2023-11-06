import time

import openai
import openai.error
import requests

import os
from logger import get_logger


story = "Tortoise embarks on an epic journey through bright and colorful scenery."
prompt_num = 6
    # - best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress
    # - masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes,
    # - best quality, masterpiece, 1boy, formal, abstract, looking at viewer, masculine, marble pattern
    # - best quality, masterpiece, 1girl, cloudy sky, dandelion, contrapposto, alternate hairstyle
examples = """
    - daytime,in an upscale restaurant, 1girl, white dress, black hair, sits at exquisitely decorated tables and eat gracefully, Ultra realistic illustration, best quality, masterpiece, hyperrealistic, 8K, summer vacation
    - 1girl, white dress, black hair, walks in the busy commercial busy street,{no car but several passers-by}，Ultra realistic illustration, best quality, masterpiece, hyperrealistic, 8K, summer vacation
    - The sun was setting on the wide beach. 1girl, white dress, black hair, glides across the beach, Ultra realistic illustration, best quality, masterpiece, hyperrealistic, 8K, summer vacation
    """

logger = get_logger()
class ChatGPTBot(object):
    def __init__(self):
        super().__init__()
        # set the default api_key
        openai.api_key = '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'
        openai.api_base = 'http://openai.greatleapai.com/v1'
        self.system_prompt_story = """
            Examples of scene discription are
            Scene1: In an upscale restaurant, a woman in a white dress sits at exquisitely decorated tables and eat gracefully
            Scene2: A women in white elegant dress walks in the busy commercial busy street
            Scene3: 一个穿长裙的美女走在繁华的大街上
            
            Write a story based on user's input with {} scene descriptions.
            You should answer in the following format:
            Scene1: xxx
            Scene2: xxx
            ...
            """
        self.system_prompt_prompt = """
            Examples of high quality prompt for text-to-image models (Stable Diffusion, midjourney or Dalle2) are
                {}
                
            Write high quality prompts for each scene description in user input.
            You should answer in English in the following format:
            Prompt1: xxx
            Prompt2: xxx
            ...
            """
    
    def extract_prompts(self, resp, type='Scene'):
        lines = resp.split('\n')
        prompts = []
        for line in lines:
            if line.strip == "" or len(line.split(':')) != 2:
                continue
            if line.startswith(type):
                prompts.append(line.split(':')[1])
        resp = '\n'.join(prompts)
        print('{}:\n{}'.format(type, prompts))
        return resp
                
    def make_story(self, keywords, FrameNum=6):        
        system_prompt = self.system_prompt_story.format(FrameNum)
        resp = self.call(system_prompt, keywords)
        stories = self.extract_prompts(resp, 'Scene')
        print('Chatgpt Resp: {}\nStroies : {}'.format(resp, stories))
        return stories
    
    def make_prompt(self, story):        
        system_prompt = self.system_prompt_prompt.format(examples)
        resp = self.call(system_prompt, story)
        prompts = self.extract_prompts(resp, 'Prompt')
        print('\nChatgpt system_prompt {} \nResp: {}\nPrompt : \n{}'.format(system_prompt, resp, prompts))
        
        return prompts
        
    def call(self, system_prompt, query):
        #try:
        
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}]
        response = openai.ChatCompletion.create(
            model='gpt-4',
            #model='gpt-3.5-turbo',
            messages=messages,
            #max_tokens=2000,
            #temperature=0.0,
            #timeout=180
            )  # 在 messages 中加入 `助手(assistant)` 的回答
        res = response.choices[0]["message"].content
        reply_content = response.choices[0]['message']['content']
        logger.info("[ChatGPT] final_reply={}".format(reply_content)) 
        print('chatgpt response:\n{}'.format(reply_content))
        return reply_content
        # except Exception as e:
        #     logger.error('Chatgpt call faile: query = {}, error={}'.format(query, e))

if __name__ == '__main__':
    bot = ChatGPTBot()
    story = bot.make_story('美女旅游')
    prompts = bot.make_prompt(story)
    