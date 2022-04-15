import os
import shutil
from enum import Enum
from pathlib import Path

import face_enhancement
import my_utils
import quality_enhancement
import scratch_detection
from deoldify import visualize

DEFAULT_INPUT_DIR = 'test_input'
DEFAULT_OUTPUT_DIR = 'test_output'

DEFAULT_GPEN_OPTIONS = {}
DEFAULT_OLDP_OPTIONS = {}

TEMP_INPUT = 'temp/input'
TEMP_OUTPUT = 'temp/output'


class RunMode(Enum):
    ENHANCE_RESTORE = 1
    RESTORE_ENHANCE = 2
    ONLY_RESTORE = 3


# Set home for CUDA
# os.environ['CUDA_HOME'] = '/usr/local/cuda/'
# os.environ['PATH'] = f'/usr/local/cuda:{os.environ["PATH"]}'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb=100"

def copy_files(files, dest_dir):
    # print(files)
    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, dest_dir)


def run(input_dir, output_dir, inpaint_scratches=False,
        colorize=False, GPU="0", sr_scale=2, hr_quality=False,
        hr_restore=False, run_mode=RunMode.ENHANCE_RESTORE):
    my_utils.remake_dir(output_dir)

    enhance_quality = True

    if inpaint_scratches:
        print(f"Running scratch detection on {input_dir}")
        input_dir, masks_dir = scratch_detection.run(
            input_dir, os.path.join(output_dir, 'scratch'), GPU=GPU, input_size="full_size")

        print(f"Running quality enhancement on {input_dir}")
        input_dir = quality_enhancement.run(
            input_dir, os.path.join(output_dir, 'quality_enh'),
            HR=hr_quality,
            masks_dir=masks_dir)
        enhance_quality = False

    if run_mode is RunMode.ENHANCE_RESTORE and enhance_quality:
        print(f"Running quality enhancement on {input_dir}")
        input_dir = quality_enhancement.run(
            input_dir, os.path.join(output_dir, 'quality_enh'),
            HR=hr_quality, test_mode="Full")

        enhance_quality = False

    print(f"Running face restoration/enhancement & super resolution on {input_dir}")
    input_dir = face_enhancement.run(
        input_dir, os.path.join(output_dir, 'face_restore'), sr_scale=sr_scale, use_cuda=not hr_restore)

    rerun_restoration = False
    if run_mode is RunMode.RESTORE_ENHANCE and enhance_quality:
        # rerun_restoration = True
        print(f"Running quality enhancement on {input_dir}")
        input_dir = quality_enhancement.run(
            input_dir, os.path.join(output_dir, 'quality_enh'),
            HR=hr_quality, test_mode="Full")

    if rerun_restoration:
        print(f"Running face restoration/enhancement & super resolution on {input_dir}")
        input_dir = face_enhancement.run(
            input_dir, os.path.join(output_dir, 'face_restore2'), sr_scale=sr_scale,
            use_cuda=not hr_restore)

    if colorize:
        output_dir = os.path.join(output_dir, 'colorization')
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            image_path = os.path.join(input_dir, filename)

            if not os.path.isfile(image_path) or os.path.splitext(filename)[-1][1:] not in ['png', 'jpg', 'jpeg']:
                # print(os.path.splitext(filename)[-1][1:])
                print(f'Skipping non-image path: {filename}')
                continue

            print(f'Processing: {filename}')

            colorizer = visualize.get_image_colorizer(artistic=True)
            result = colorizer.get_transformed_image(
                Path(image_path),
                render_factor=30,
                post_process=True,
                watermarked=False)

            if result is not None:
                result.save(os.path.join(output_dir, filename), quality=95)
                result.close()
            else:
                print(f'Colorization failed for {image_path}')

        input_dir = output_dir

    print(input_dir)
    return input_dir



def main():
    # run('sample_image', 'output/out2', sr_scale=4, run_mode=RunMode.ENHANCE_RESTORE)
    run('sample_image', 'output/out1', sr_scale=4, run_mode=RunMode.RESTORE_ENHANCE, colorize=True, hr_restore=True)
    # run('sample_image', 'output/out3', sr_scale=4, run_mode=RunMode.ONLY_RESTORE)


if __name__ == '__main__':
    main()
