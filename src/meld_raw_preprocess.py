import subprocess
import os
import re

from moviepy.video.io.VideoFileClip import VideoFileClip

video_path = 'F:\BaiduNetdiskDownload\MELD.Raw\MELD.Raw/test\output_repeated_splits_test'
audio_path = 'F:\ERC_CODE\meld_feature\wav_meld_test'


files = os.listdir(video_path)

# 遍历输入文件夹中的所有文件
for idx, file_name in enumerate(files):
    pattern = r'^dia\d+_utt\d+\.mp4$'
    match = re.match(pattern, file_name)
    if match and file_name.endswith('.mp4'):
        # 构建输入文件的完整路径
        input_file = os.path.join(video_path , file_name)

        # 构建输出文件的完整路径
        output_file = os.path.join(audio_path, file_name[:-4]+'.wav')

        video = VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(output_file)
        print('success! audio save at'+output_file)
