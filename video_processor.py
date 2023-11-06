import os
from glob import glob
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, TextClip, concatenate_videoclips, CompositeVideoClip


class MoiveEditor(object):
    def __init__(self):
        return
    
    def save_video(self, video_file, path):
        if os.path.exists(path):
            print(f"{path} exists.")
            #return
        video_file.write_videofile(path)
        return
    
    def merge_videos(self, clips):
        final_clip = concatenate_videoclips(clips)
        return final_clip
    
    def video_edit(self, video_path, speed_factor=0.5, text=None, audio_path=None):
        print('video_path:{}\ntext:{}\naudio_path:{}'.format(video_path, text, audio_path))
        video = VideoFileClip(video_path)
        if audio_path:
            audio = AudioFileClip(audio_path)
            video = video.set_audio(audio)
        if speed_factor != 1.0:
            video = video.speedx(factor=speed_factor)  # Slow down the video by half    
        if text:
            w, h = video.size
            screensize = (w, int(h/5))
            print(screensize)
            text_clip = TextClip(text, fontsize=20, font='gkai00mp.ttf', color="white", size=screensize, method='caption')
            text_clip = text_clip.set_pos('bottom').set_duration(video.duration)
            video = CompositeVideoClip([video, text_clip])
        return video
    
    def process(self, paths, stories):
        print(paths, stories)
        if stories:
            clips = [self.video_edit(path, 0.5, prompt) for path, prompt in zip(paths, stories)]
        else:
            clips = [self.video_edit(path, 0.5, None) for path in paths]
        return clips

    def merge(self,
        path,
        prompt,
        seed,
        image):
        videos = [p for p in glob(os.path.join(path, "*")) if (not p.endswith('merge.mp4') and '{}'.format(seed) in p)]
        videos.sort()

        outfile_path = os.path.join(path, f"{prompt}_{seed}_merge.mp4")
        clips = self.process(videos, None)
        print('image = {}'.format(image))
        if image is not None:
            image = ImageClip(image).set_duration(3).set_fps(24)
            clips.append(image)
        if len(clips) > 0:
            final_video = concatenate_videoclips(clips)
            self.save_video(final_video, outfile_path)

        print(videos)
        print(outfile_path)
        return outfile_path

if __name__ == '__main__':
    edit = MoiveEditor()
    base_path = '/data/huyong/AnimateDiffV0/samples/Gradio-2023-10-16T06-59-26/sample/'
    edit.merge(base_path, '我们的作', -1, base_path + 'jingdianMilk.jpeg')
#     prompts_str ="""我们的作品
# Highest quality, stunning artwork, single female figure, exterior location, abundant field, golden sunset, colourful wildflower bouquet
# Best quality, masterpiece, 1boy, reading, book, outdoors, under oak tree, autumn
# Highest quality, magnificent portrayal, solo male, engaging with literature, exterior location, beneath towering oak, fall season
# Best quality, masterpiece, 1girl, looking at stars, snowy mountaintop, night time
# Highest quality, beautiful artwork, individual woman, stargazing, atop snow-covered peak, darkness of night
# Best quality, masterpiece, group of people, indoors, laughing, eating hotpot, table, cozy room
# Highest quality, enchanting portrayal, group interaction, interior setting, joyous laughter, consuming hotpot, cozy atmosphere"""
#     prompts = prompts_str.split('\n')
#     print(prompts)
#     paths = [base_path+'{}.mp4'.format(x) for x in range(7)]
#     clips = [edit.video_edit(path, 0.5, prompt) for path, prompt in zip(paths, prompts)]
#     final_video = edit.merge_videos(clips)
#     edit.save_video(final_video, base_path + 'output14.mp4')
