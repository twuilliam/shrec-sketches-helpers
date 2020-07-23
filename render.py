import os
import sys
from BlenderPhong import phong


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]

    models_list = str(argv[0])
    out_dir = str(argv[1])

    with open(models_list) as f:
        models = f.read().splitlines()

    dataset = models[0].split('/')[0]

    if dataset == 'SHREC13':
        base = os.path.join(out_dir, 'SHREC13', 'SHREC13_SBR_TARGET_MODELS_IMGS')
    elif dataset == 'SHREC14':
        base = os.path.join(out_dir, 'SHREC14', 'SHREC14LSSTB_TARGET_MODELS_IMGS')

    image_dir = os.path.join(base, 'orig')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # blender has no native support for off files
    phong.install_off_addon()

    phong.init_camera()
    phong.fix_camera_to_origin()

    for model in models:
        model_path = os.path.join(out_dir, model)
        phong.do_model(model_path, image_dir)


if __name__ == '__main__':
    main()
