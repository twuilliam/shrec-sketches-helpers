import os
import argparse
import gensim.downloader as api
import numpy as np
import pandas as pd


# manually curated names to match word2vec entries
NAMES = {'pipe_for_smoking': ['pipe', 'smoking'],
         'parking_meter': ['metered_parking'],
         'door_handle': ['doorhandle'],
         'bear_animal': ['bear'],
         'race_car': ['racecar'],
         'axe': ['ax'],
         'tablelamp': ['table', 'lamp'],
         'beer_mug': ['beer_mugs'],
         'flower_with_stem': ['flower', 'stem'],
         'wrist_watch': ['wristwatch']}


def config():
    parser = argparse.ArgumentParser(description='SHREC meta')
    parser.add_argument('--dataset', metavar='D', default='14',
                        choices=['13', '14'],
                        help='dataset')
    args = parser.parse_args()
    return args


def get_vector_names(classnames):
    print('Loading word2vec...')
    model = api.load("word2vec-google-news-300")

    wv = {}
    for cls in classnames:
        print(cls)
        tmp = cls.replace('-', '_')
        try:
            vec = model.get_vector(tmp)
        except:
            if tmp in NAMES:
                vec = np.mean([model.get_vector(w) for w in NAMES[tmp]], axis=0)
            else:
                vec = np.mean([model.get_vector(w) for w in tmp.split('_')], axis=0)

        wv[cls] = vec
    return wv


def main():
    args = config()

    if args.dataset == '13':
        save_dir = os.path.join('labels', 'SHREC13')
        df = pd.read_hdf(os.path.join(save_dir, 'sk_orig.hdf5'))
    elif args.dataset == '14':
        save_dir = os.path.join('labels', 'SHREC14')
        df = pd.read_hdf(os.path.join(save_dir, 'sk_orig.hdf5'))

    classnames = df['cat'].unique()
    wv = get_vector_names(classnames)

    np.savez(os.path.join(save_dir, 'w2v.npz'), wv=wv)

    if args.dataset == '14':
        save_dir = os.path.join('labels', 'PART-SHREC14')
        df = pd.read_hdf(os.path.join(save_dir, 'sk_orig.hdf5'))
        classnames = df['cat'].unique()

        part_wv = {}
        for cls in classnames:
            part_wv[cls] = wv[cls]

        np.savez(os.path.join(save_dir, 'w2v.npz'), wv=part_wv)


if __name__ == "__main__":
    main()
