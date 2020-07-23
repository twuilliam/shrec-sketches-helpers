import argparse
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from PIL import Image, ImageOps


DESIRED_SIZE = 256


def config():
    parser = argparse.ArgumentParser(description='SHREC meta')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='data folder path')
    parser.add_argument('--dataset',
                        required=True,
                        choices=['13', '14'],
                        help='dataset')
    args = parser.parse_args()
    return args


def transform_im(im):
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    # resize
    old_size = np.asarray(im.size)
    ratio = float(DESIRED_SIZE) / max(old_size)
    new_size = map(int, old_size * ratio)
    im = im.resize(new_size, Image.ANTIALIAS)

    return im


def process_im(path):
    im = Image.open(path)
    im = transform_im(im)

    # saving
    im_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    if 'SHREC13' in path:
        split = im_dir.split(os.path.sep)[-1]
        clsname = im_dir.split(os.path.sep)[-2]
        if split == 'test':
            save_dir = os.path.join(*im_dir.split(os.path.sep)[:-3])
        elif split == 'train':
            save_dir = os.path.join(*im_dir.split(os.path.sep)[:-4])
        save_dir = os.path.join(
            os.path.sep, save_dir,
            'SHREC13_SBR_SKETCHES_RESIZED', clsname, split)
    elif 'SHREC14' in path:
        split = im_dir.split(os.path.sep)[-1]
        clsname = im_dir.split(os.path.sep)[-2]

        save_dir = os.path.join(*im_dir.split(os.path.sep)[:-4])
        save_dir = os.path.join(
            os.path.sep, save_dir,
            'SHREC14LSSTB_SKETCHES_RESIZED', clsname, split)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    im.save(os.path.join(save_dir, fname), quality=95)


def process_idx(idx):
    for i in idx:
        process_im(i)


def worker(q, idx):
    q.put(process_idx(idx))


def domulti(ncpus, paths):
    n = len(paths)

    q = mp.Queue()
    processes = []
    for i in range(ncpus):
        lower = int((i) * n / (ncpus))
        upper = int((i + 1) * n / (ncpus))
        processes.append(mp.Process(target=worker,
                                    args=(q, paths[lower:upper])))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def transform_path(df):
    paths = df.index

    new_paths = []
    for p in paths:
        dataset = p.split(os.path.sep)[0]
        tmp = os.path.join(*p.split(os.path.sep)[-3:])
        if dataset == 'SHREC13':
            new_paths.append(
                os.path.join(dataset, 'SHREC13_SBR_SKETCHES_RESIZED', tmp))
        elif dataset == 'SHREC14':
            new_paths.append(
                os.path.join(dataset, 'SHREC14LSSTB_SKETCHES_RESIZED', tmp))
    df.index = new_paths
    return df


def main():
    args = config()

    if args.dataset == '13':
        df = pd.read_hdf(os.path.join('labels', 'SHREC13', 'sk_orig.hdf5'))
    if args.dataset == '14':
        df = pd.read_hdf(os.path.join('labels', 'SHREC14', 'sk_orig.hdf5'))

    paths = df.index.values
    all_paths = []
    for p in paths:
        all_paths.append(os.path.join(args.data_dir, p))
    domulti(10, all_paths)
    df_resized = transform_path(df)
    if args.dataset == '13':
        df_resized.to_hdf(os.path.join('labels', 'SHREC13', 'sk_resized.hdf5'), 'sk')
    if args.dataset == '14':
        df_resized.to_hdf(os.path.join('labels', 'SHREC14', 'sk_resized.hdf5'), 'sk')

        df_part = pd.read_hdf(os.path.join('labels', 'PART-SHREC14', 'sk_orig.hdf5'))
        df_part_resized = transform_path(df_part)
        df_part_resized.to_hdf(os.path.join('labels', 'PART-SHREC14', 'sk_resized.hdf5'), 'sk')


if __name__ == '__main__':
    main()
