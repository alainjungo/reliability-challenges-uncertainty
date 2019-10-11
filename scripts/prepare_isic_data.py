import os

from PIL import Image

import common.data.collector as collect
import common.utils.filehelper as fh
import rechun.directories as dirs


def main():
    out_dir = dirs.ISIC_PREPROCESSED_DIR
    fh.create_and_clean_dir(out_dir)

    process_data(out_dir, dirs.ISIC_ORIG_TRAIN_DATA_DIR)
    process_data(out_dir, dirs.ISIC_ORIG_VALID_DATA_DIR)
    process_data(out_dir, dirs.ISIC_ORIG_TEST_DATA_DIR)


def process_data(out_dir: str, in_dir_with_task_prefix: str):
    print('Process: {}'.format(os.path.basename(in_dir_with_task_prefix)))
    collector = collect.IsicCollector(in_dir_with_task_prefix, with_super_pixels=True)
    img_dir, label_dir = collector.get_img_and_label_dirs()

    out_img_dir = os.path.join(out_dir, os.path.basename(img_dir))
    os.makedirs(out_img_dir)
    out_label_dir = os.path.join(out_dir, os.path.basename(label_dir))
    os.makedirs(out_label_dir)

    new_size = (192, 256)  # h, w
    new_size = new_size[::-1]  # since pil has wxh

    for i, subject_file in enumerate(collector.subject_files):
        print('[{}/{}] {}'.format(i+1, len(collector.subject_files), subject_file.subject))
        files = subject_file.get_all_files()

        img = Image.open(files['image'])
        img = img.resize(new_size, Image.BILINEAR)
        img.save(os.path.join(out_img_dir, os.path.basename(files['image'])))

        label = Image.open(files['gt'])
        label = label.resize(new_size, Image.NEAREST)
        label.save(os.path.join(out_label_dir, os.path.basename(files['gt'])))

        superpixel = Image.open(files['superpixel'])
        superpixel = superpixel.resize(new_size, Image.NEAREST)  # type: Image
        superpixel.save(os.path.join(out_img_dir, os.path.basename(files['superpixel'])))


if __name__ == '__main__':
    main()
