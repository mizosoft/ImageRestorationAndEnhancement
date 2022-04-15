import os
from pathlib import Path

from deoldify import visualize

input_path = 'colorize_test/restore_enhance.jpeg'
output_path = 'colorize_test/shika2/restore_enhance.png'
render_factor = 35

colorizer = visualize.get_image_colorizer(artistic=False)
result = colorizer.get_transformed_image(Path(input_path), render_factor=render_factor, post_process=True, watermarked=False)
if result is not None:
    result.save(output_path, quality=100)
    result.close()
else:
    print('shashanya')
