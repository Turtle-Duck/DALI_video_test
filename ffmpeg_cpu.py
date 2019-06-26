# python 3.4

import imageio  # version 2.4.1

path = 'video_data/7275702640_b.mp4'
vid = imageio.get_reader(path, 'ffmpeg')
index = 0  # The result I posted in the first comment is index 4.
frame = vid.get_data(0)[:, :, [2, 1, 0]]

print(frame)



