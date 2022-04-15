import os
from pathlib import Path

from deoldify import visualize

input_path = 'colorize_test/b.png'
output_path = 'colorize_test/out2/b.png'
render_factor = 30

colorizer = visualize.get_image_colorizer(artistic=False)
result = colorizer.get_transformed_image(Path(input_path), render_factor=render_factor, post_process=True, watermarked=False)
if result is not None:
    result.save(output_path, quality=95)
    result.close()
else:
    print('shashanya')




