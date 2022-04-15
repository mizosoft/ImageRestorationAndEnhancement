import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

add_path(os.path.join(this_dir, 'face_detect'))
add_path(os.path.join(this_dir, 'face_parse'))
add_path(os.path.join(this_dir, 'face_model'))
add_path(os.path.join(this_dir, 'sr_model'))
add_path(os.path.join(this_dir, 'training'))
add_path(os.path.join(this_dir, 'training/loss'))
add_path(os.path.join(this_dir, 'training/data_loader'))
# add_path(os.path.join(this_dir, 'quality'))
