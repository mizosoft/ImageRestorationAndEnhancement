# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import gc
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
from PIL.Image import Resampling

from quality.detection_models import networks
from quality.detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(img, full_size, method=Resampling.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")

#     parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
def run(input_dir, mask_output_dir, GPU, input_size="scale_256"):
    print("initializing the dataloader")

    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    ## load model
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/detection/FT_Epoch_latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("model weights loaded")

    GPU = int(GPU)
    if GPU >= 0:
        # print(os.environ)
        model.to(GPU)
    else:
        model.cpu()
    model.eval()

    ## dataloader and transformation
    print("directory of testing image: " + input_dir)
    imagelist = os.listdir(input_dir)
    imagelist.sort()
    total_iter = 0

    save_url = os.path.join(mask_output_dir)
    mkdir_if_not(save_url)

    rescaled_dir = os.path.join(save_url, "rescaled")
    mask_output_dir = os.path.join(save_url, "mask")
    # blend_output_dir=os.path.join(save_url, 'blend_output')
    mkdir_if_not(rescaled_dir)
    mkdir_if_not(mask_output_dir)
    # mkdir_if_not(blend_output_dir)

    idx = 0

    for image_file in imagelist:

        idx += 1

        print("processing", image_file)

        image_name, _ = os.path.splitext(image_file)

        scratch_file = os.path.join(input_dir, image_file)
        if not os.path.isfile(scratch_file):
            print("Skipping non-file %s" % image_file)
            continue
        scratch_image = Image.open(scratch_file).convert("RGB")
        w, h = scratch_image.size

        transformed_image_PIL = data_transforms(scratch_image, input_size)
        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = scale_tensor(scratch_image)

        if GPU >= 0:
            scratch_image_scale = scratch_image_scale.to(GPU)
        else:
            scratch_image_scale = scratch_image_scale.cpu()
        with torch.no_grad():
            P = torch.sigmoid(model(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")

        tv.utils.save_image(
            (P >= 0.4).float(),
            os.path.join(
                mask_output_dir,
                image_name + ".png",
            ),
            nrow=1,
            padding=0,
            normalize=True,
        )
        transformed_image_PIL.save(os.path.join(rescaled_dir, image_name + ".png"))
        gc.collect()
        torch.cuda.empty_cache()

    return rescaled_dir, mask_output_dir
