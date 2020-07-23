import argparse
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from PIL import Image, ImageOps


DESIRED_SIZE = 256
MARGIN = 100


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


def transform_im(im, margin=0):
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    # get roi coordinates
    mask = np.asarray(im) != 0

    min_w = np.min(np.argwhere(mask)[:, 1]) - margin
    max_w = np.max(np.argwhere(mask)[:, 1]) + margin

    min_h = np.min(np.argwhere(mask)[:, 0]) - margin
    max_h = np.max(np.argwhere(mask)[:, 0]) + margin

    bbox = [min_w, min_h, max_w, max_h]

    # cropping
    im = im.crop(bbox)

    # resize
    old_size = np.asarray(im.size)
    ratio = float(DESIRED_SIZE) / max(old_size)
    new_size = map(int, old_size * ratio)
    im = im.resize(new_size, Image.ANTIALIAS)

    # padding
    delta_w = DESIRED_SIZE - new_size[0]
    delta_h = DESIRED_SIZE - new_size[1]
    padding = (delta_w // 2, delta_h // 2,
               delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    black = (0, 0, 0)
    im = ImageOps.expand(im, padding, black)

    return im


def process_im(path):
    im = Image.open(path)
    im = transform_im(im, margin=MARGIN)

    # saving
    im_dir = os.path.dirname(path)
    model = im_dir.split(os.path.sep)[-1]
    fname = os.path.basename(path)
    save_dir = os.path.join(*im_dir.split(os.path.sep)[:-2])
    save_dir = os.path.join(os.path.sep, save_dir, 'resized', model)
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


def get_img_paths(args):
    if args.dataset == '13':
        base = os.path.join(args.data_dir, 'SHREC13', 'SHREC13_SBR_TARGET_MODELS_IMGS')

    elif args.dataset == '14':
        base = os.path.join(args.data_dir, 'SHREC14', 'SHREC14LSSTB_TARGET_MODELS_IMGS')

    image_dir = os.path.join(base, 'orig')

    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f[-3:] == 'png':
                paths.append(os.path.join(root, f))
    return paths


def save_img_path(args):
    if args.dataset == '13':
        base = os.path.join('SHREC13', 'SHREC13_SBR_TARGET_MODELS_IMGS')
        df_cad_dir = os.path.join('labels', 'SHREC13')
    elif args.dataset == '14':
        base = os.path.join('SHREC14', 'SHREC14LSSTB_TARGET_MODELS_IMGS')
        df_cad_dir = os.path.join('labels', 'SHREC14')

    image_dir = os.path.join(base, 'resized')

    ids = []
    paths = []
    views = []
    for root, _, files in os.walk(os.path.join(args.data_dir, image_dir)):
        for f in files:
            if f[-3:] == 'png':
                folder = root.split(os.path.sep)[-1]
                paths.append(os.path.join(image_dir, folder, f))
                ids.append(os.path.splitext(f)[0].split('.')[0])
                views.append(os.path.splitext(f)[0].split('.')[1])

    df = pd.DataFrame({'id': ids, 'views': views}, index=paths)

    df_cad = pd.read_hdf(os.path.join(df_cad_dir, 'cad_orig.hdf5'))
    cat = df_cad['cat'].unique()
    for c in cat:
        to_select = df_cad.loc[df_cad['cat'] == c, 'id'].unique()
        df.loc[df['id'].isin(to_select), 'cat'] = c

    df.to_hdf(os.path.join(df_cad_dir, 'cad_img_resized.hdf5'), 'cad')

    if args.dataset == '14':
        df_cad_dir = os.path.join('labels', 'PART-SHREC14')
        df_cad = pd.read_hdf(os.path.join(df_cad_dir, 'cad_orig.hdf5'))

        # select ids
        to_select = df_cad['id'].unique()
        part_df = df.loc[df['id'].isin(to_select)].copy()

        # parse train-test split
        to_select = df_cad.loc[df_cad['split'] == 'train', 'id'].unique()
        part_df.loc[part_df['id'].isin(to_select), 'split'] = 'train'

        to_select = df_cad.loc[df_cad['split'] == 'test', 'id'].unique()
        part_df.loc[part_df['id'].isin(to_select), 'split'] = 'test'

        part_df.to_hdf(os.path.join(df_cad_dir, 'cad_img_resized.hdf5'), 'cad')


def main():
    args = config()
    paths = get_img_paths(args)
    domulti(10, paths)
    save_img_path(args)


if __name__ == '__main__':
    main()
