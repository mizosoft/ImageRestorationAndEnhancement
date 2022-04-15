import glob
import os

import cv2
import numpy as np
import torch.cuda


def process(processor, indir, outdir, task, aligned=False, save_face=False):
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)

        img = cv2.imread(file, cv2.IMREAD_COLOR)  # BGR
        if not isinstance(img, np.ndarray):
            print(filename, 'error')
            continue
        # img = cv2.resize(img, (0,0), fx=2, fy=2) # optional

        print(f'Processing: {n}: {filename}')

        img_out, orig_faces, enhanced_faces = processor.process(img, aligned=aligned)

        # img = cv2.resize(img, img_out.shape[:2][::-1])
        # cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_COMP.jpg'),
        #             np.hstack((img, img_out)))
        # cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_GPEN.jpg'), img_out)

        cv2.imwrite(os.path.join(outdir, filename), img_out)

        # print(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_GPEN.jpg'))

        if save_face:
            for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                of = cv2.resize(of, ef.shape[:2])
                cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_face%02d' % m + '.jpg'),
                            np.hstack((of, ef)))

        torch.cuda.empty_cache()

