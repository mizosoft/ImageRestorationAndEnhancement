from io import BytesIO

import main
import os
import cv2
import PySimpleGUI as sg

import my_utils
import shutil
from PIL import Image


def modify(image_file, colorize=False, scratches=False):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_folder", type=str, default="input", help="Input folder")
    # parser.add_argument("--output_folder", type=str, default="output/soka", help="Output folder")
    # parser.add_argument("--run_mode", type=int, default=1, choices=range(1, 5), help= "Setting run mode, 1-> ENHANCE_RESTORE 2->RESTORE_ENHANCE 3->RESTORE_ONLY 4->ENHANCE_ONLY")
    # parser.add_argument("--sr_scale", type=int, default=4)
    # parser.add_argument("--hr_quality", action='store_true')
    # parser.add_argument("--hr_restore", action='store_true')
    # args = parser.parse_args()

    print(f'handling input: {image_file}')

    image_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    input_dir = os.path.join(image_dir, 'temp')
    input_img_file = os.path.join(input_dir, image_name)

    my_utils.remake_dir(input_dir)
    shutil.copy2(image_file, input_img_file)

    output_dir = main.run(input_dir=input_dir, output_dir='output/', sr_scale=2, run_mode=main.RunMode.ENHANCE_RESTORE,
                          hr_quality=True, hr_restore=True, use_gpu=False, colorize=colorize,
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
                  [sg.Button('Modify Photo', key='-MPHOTO-'), sg.Button('Exit')],
                  [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],
                  [sg.Checkbox('Colorize', default=False, key='-COLORIZE-'),
                   sg.Checkbox('Inpaint Scratches', default=False, key='-SCRATCHES-')]]

    # ----- Full layout -----
    layout = [[sg.VSeperator(), sg.Column(images_col)]]

    # ----- Make the window -----
    window = sg.Window('Image Restoration & Enhancement', layout, grab_anywhere=True)

    # ----- Run the Event Loop -----
    prev_filename = None
    colorize = False
    scratches = False
    filename = None
    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break

        elif event == '-MPHOTO-':
            if filename is None:
                continue

            try:
                # n1 = filename.split("/")[-2]
                # n2 = filename.split("/")[-3]
                # n3 = filename.split("/")[-1]
                # filename= str(f"./{n2}/{n1}")
                output_img = modify(filename, colorize=values['-COLORIZE-'], scratches=values['-SCRATCHES-'])

                # global f_image
                # f_image = f'./output/final_output/{os.path.basename}'
                image = Image.open(output_img)

                orig_image = Image.open(filename)
                image = image.resize(orig_image.size)

                window['-OUT-'].update(image_to_data(image))

            except Exception as e:
                print('problem while modifying image')
                print(e.with_traceback(None))
                continue

        elif event == '-IN FILE-':  # A single filename was chosen
            filename = values['-IN FILE-']
            if filename != prev_filename:
                prev_filename = filename
                try:
                    image = cv2.imread(filename)
                    window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
                except:
                    continue

    # ----- Exit program -----
    window.close()


if __name__ == '__main__':
    gpu_main()
