# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import shutil
import sys
from subprocess import call
import my_cmd



def run(*,
        input_folder='./test_images/old',
        output_folder='./output',
        GPU='-1',
        inpaint_scratches=False,
        HR=False):
    gpu1 = GPU

    # resolve relative paths before changing directory
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ## Stage 1: Overall Quality Improve
    # print("Running Stage 1: Overall restoration")

    stage_1_input_dir = input_folder
    stage_1_output_dir = os.path.join(output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    if inpaint_scratches:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        stage_1_command_1 = (
                "python detection.py --test_path "
                + stage_1_input_dir
                + " --output_dir "
                + mask_dir
                + " --input_size full_size"
                + " --GPU "
                + gpu1
        )

        if HR:
            HR_suffix = " --HR"
        else:
            HR_suffix = ""

        stage_1_command_2 = (
                "python test.py --Scratch_and_Quality_restore --test_input "
                + new_input
                + " --test_mask "
                + new_mask
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1 + HR_suffix
        )

        cmd.run(stage_1_command_1)
        cmd.run(stage_1_command_2)
    else:
        stage_1_command = (
                "python test.py --test_mode Full --Quality_restore --test_input "
                + stage_1_input_dir
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1
        )
        cmd.run(stage_1_command)


    ## Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...")
    print("\n")


if __name__ == '__main__':
    shutil.rmtree('test_output', ignore_errors=True)

    run(input_folder='test_images/old', output_folder='test_output', GPU='0')
