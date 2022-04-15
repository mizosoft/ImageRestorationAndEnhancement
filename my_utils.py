import os
import shutil
import sys
import cv2
import __init_paths
import PIL
from PIL import Image
from face_detect.retinaface_detection import  RetinaFaceDetection


def clear(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.islink(path):
        os.unlink(path)


def remake_dir(path):
    clear(path)
    os.makedirs(path, exist_ok=True)


def has_high_res(input_dir):
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            if not os.path.isfile(file_path):
                continue

            img = Image.open(file_path)
            width, height = img.size
            if width * height > 300 ** 2:
                return True
        except PIL.UnidentifiedImageError:
            continue
    return False


def partition_wface(input_dir):
    temp_dir = os.path.join(input_dir, 'temp')
    wface = os.path.join(temp_dir, 'wface')
    woface = os.path.join(temp_dir, 'woface')

    remake_dir(wface)
    remake_dir(woface)

    for file in os.listdir(input_dir):
        try:
            file_path = os.path.join(input_dir, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            facebs, landms = RetinaFaceDetection(base_dir='./').detect(img)
            if len(facebs) | len(landms) > 0:
                shutil.copy(file_path, os.path.join(wface, file))
            else:
                shutil.copy(file_path, os.path.join(woface, file))
        except PIL.UnidentifiedImageError:
            continue
    return wface, woface


# partition_wface('partition_test')
