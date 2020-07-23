import argparse
import os
import numpy as np
import pandas as pd


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


def get_df_sketches(data_dir, sk_path):
    split = []
    cat = []
    paths = []
    ids = []
    for root, _, files in os.walk(os.path.join(data_dir, sk_path)):
        for f in files:
            if f[-3:] == 'png':
                split.append(root.split(os.path.sep)[-1])
                cat.append(root.split(os.path.sep)[-2])
                ids.append(os.path.splitext(f)[0])
                paths.append(os.path.join(sk_path, cat[-1], split[-1], f))

    df = pd.DataFrame(data={'cat': cat, 'split': split, 'id': ids},
                      index=paths)
    return df


def get_df_models(data_dir, cad_anno, cad_path):
    # read meta file
    fpath = os.path.join(data_dir, cad_anno)

    with open(fpath, 'r') as f:
        content = f.readlines()

    labels = {}
    current_cat = ''
    for line in content[3:]:
        line = line.strip('\r\n')
        line = line.strip('\t')
        line = line.strip()
        if len(line.split()) == 3:
            current_cat = line.split()[0]
        elif line != '':
            labels[line] = current_cat

    # read model folder
    cat = []
    ids = []
    paths = []
    for root, _, files in os.walk(os.path.join(data_dir, cad_path)):
        for f in files:
            if f[-3:] == 'off':
                ids.append(os.path.splitext(f)[0])
                cat.append(labels[ids[-1][1:]])
                paths.append(os.path.join(cad_path, f))

    df = pd.DataFrame(data={'cat': cat, 'id': ids},
                      index=paths)
    return df


def split_models(df_sk, df_cad):
    vv, cc = np.unique(df_cad['cat'], return_counts=True)
    coi = vv[cc > 50]
    n_coi = cc[cc > 50]

    new_df_sk = df_sk.loc[df_sk['cat'].isin(coi)].copy()
    new_df_cad = df_cad.loc[df_cad['cat'].isin(coi)].copy()

    # randomly split instances
    np.random.seed(1234)
    new_df_cad.loc[:, 'split'] = 'train'
    for c, n in zip(coi, n_coi):
        to_select = int(np.floor(n * 0.2))
        subset = new_df_cad.loc[new_df_cad['cat'] == c, 'id']
        id_to_select = np.random.choice(subset, size=to_select, replace=False)
        new_df_cad.loc[new_df_cad['id'].isin(id_to_select), 'split'] = 'test'
    return new_df_sk, new_df_cad


def main():
    args = config()

    if args.dataset == '14':
        base = 'SHREC14'

        # get sketch labels
        sk_path = os.path.join(base, 'SHREC14LSSTB_SKETCHES', 'SHREC14LSSTB_SKETCHES')
        df_sk = get_df_sketches(args.data_dir, sk_path)

        cad_path = os.path.join(base, 'SHREC14LSSTB_TARGET_MODELS')
        eval_path = os.path.join(base, 'SHREC14_Sketch_Evaluation_CVIU')
        cad_anno = os.path.join(eval_path, 'SHREC14_SBR_Model.cla')

    elif args.dataset == '13':
        base = 'SHREC13'

        # get sketch labels (in two different folders)
        sk_path_tr = os.path.join(
            base, 'SHREC13_SBR_TRAINING_SKETCHES', 'SHREC13_SBR_TRAINING_SKETCHES')
        sk_path_te = os.path.join(
            base, 'SHREC13_SBR_TESTING_SKETCHES')
        tmp1 = get_df_sketches(args.data_dir, sk_path_tr)
        tmp1['split'] = 'train'
        tmp2 = get_df_sketches(args.data_dir, sk_path_te)
        tmp2['split'] = 'test'
        df_sk = pd.concat([tmp1, tmp2])

        # get cad labels
        cad_path = os.path.join(base, 'SHREC13_SBR_TARGET_MODELS', 'models')
        eval_path = os.path.join(base, 'SHREC2013_Sketch_Evaluation')
        cad_anno = os.path.join(eval_path, 'SHREC13_SBR_Model.cla')

    # get cad labels
    df_cad = get_df_models(args.data_dir, cad_anno, cad_path)

    save_dir = os.path.join('labels', base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_sk.to_hdf(os.path.join(save_dir, 'sk_orig.hdf5'), 'sk')
    df_cad.to_hdf(os.path.join(save_dir, 'cad_orig.hdf5'), 'cad')

    with open(os.path.join(save_dir, 'cad.txt'), 'w') as f:
        for item in df_cad.index:
            f.write('%s\n' % item)

    if args.dataset == '14':
        # split between train and test cad models
        # following Qi et al BMVC 2018
        new_df_sk, new_df_cad = split_models(df_sk, df_cad)

        save_dir = os.path.join('labels', 'PART-' + base)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        new_df_sk.to_hdf(os.path.join(save_dir, 'sk_orig.hdf5'), 'sk')
        new_df_cad.to_hdf(os.path.join(save_dir, 'cad_orig.hdf5'), 'cad')


if __name__ == "__main__":
    main()
