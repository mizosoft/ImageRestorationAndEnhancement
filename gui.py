from io import BytesIO

import main
import os
import cv2
import PySimpleGUI as sg

import my_utils
import shutil
from PIL import Image


def modify(image_file, colorize=False, scratches=False):
    print(f'handling input: {image_file}')

    image_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    input_dir = os.path.join(image_dir, 'temp')
    input_img_file = os.path.join(input_dir, image_name)

    my_utils.remake_dir(input_dir)
    shutil.copy2(image_file, input_img_file)

    output_dir = main.run(input_dir=input_dir, output_dir='output/', sr_scale=2, run_mode=main.RunMode.ENHANCE_RESTORE,
                          hr_quality=True, hr_restore=False, use_gpu=False, colorize=colorize,
                          inpaint_scratches=scratches)

    img_name, _ = os.path.splitext(image_name)
    for output_file in os.listdir(output_dir):
        output_name, _ = os.path.splitext(output_file)
        if img_name == output_name:
            output_img = os.path.join(output_dir, output_file)
            print(f'Finished, output: {output_img}')
            return output_img

    raise ValueError("couldn't find output image")


def image_to_data(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return data


def gpu_main():
    # run('sample_image', 'output/out2', sr_scale=4, run_mode=RunMode.ENHANCE_RESTORE)
    # run('input', 'output/scratchbob', sr_scale=2, run_mode=RunMode.RESTORE_ENHANCE, colorize=False, hr_restore=True)
    # run('sample_image', 'output/out3', sr_scale=4, run_mode=RunMode.ONLY_RESTORE)

    images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
                  [sg.Button('Modify Photo', key='-MPHOTO-'), sg.Button('Exit'),
                   sg.Checkbox('Colorize', default=False, key='-COLORIZE-'),
                   sg.Checkbox('Inpaint Scratches', default=False, key='-SCRATCHES-')],
                  [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')]]

    layout = [[sg.VSeperator(), sg.Column(images_col)]]

    window = sg.Window('Image Restoration & Enhancement', layout, grab_anywhere=True)

    prev_img_file = None
    img_file = None
    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break

        elif event == '-MPHOTO-':
            if img_file is None:
                continue

            try:
                output_img_file = modify(img_file, colorize=values['-COLORIZE-'], scratches=values['-SCRATCHES-'])

                image = Image.open(output_img_file)
                orig_image = Image.open(img_file)
                image = image.resize(orig_image.size)

                window['-OUT-'].update(image_to_data(image))

            except Exception as e:
                print('problem while modifying image')
                print(e.with_traceback(None))
                continue

        elif event == '-IN FILE-':  # A single filename was chosen
            img_file = values['-IN FILE-']
            if img_file != prev_img_file:
                prev_img_file = img_file
                try:
                    image = Image.open(img_file)
                    window['-IN-'].update(image_to_data(image))

                except Exception as e:
                    print('problem while reading image')
                    print(e.with_traceback(None))
                    continue

    window.close()


if __name__ == '__main__':
    gpu_main()
