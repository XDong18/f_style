import os
import os.path as osp
import json

if __name__ == "__main__":
    root_dir = '/shared/xudongliu/bdd100k/seg/seg'
    info_fn = osp.join(root_dir, 'info.json')
    info_data = {
    "mean": [0.2787, 0.2926, 0.2899],
    "std": [0.2474, 0.2653, 0.2760]
    }
    with open(info_fn, 'w') as f:
        json.dump(info_data, f)
    
    train_dir = osp.join(root_dir, 'images/train/')
    val_dir = osp.join(root_dir, 'images/val/')
    test_dir = osp.join(root_dir, 'images/test/')

    train_fn =osp.join(root_dir, 'train_images.txt')
    val_fn =osp.join(root_dir, 'val_images.txt')
    test_fn =osp.join(root_dir, 'test_images.txt')

    img_list_train = sorted(os.listdir(train_dir))
    img_list_val = sorted(os.listdir(val_dir))
    img_list_test = sorted(os.listdir(test_dir))

    with open(train_fn, 'w') as f:
        for i, img_fn in enumerate(img_list_train):
            if i < len(img_list_train)-1:
                f.write(osp.join('images/train/', img_fn) + '\n')
            else:
                f.write(osp.join('images/train/', img_fn))

    with open(val_fn, 'w') as f:
        for i, img_fn in enumerate(img_list_val):
            if i < len(img_list_val)-1:
                f.write(osp.join('images/val/', img_fn) + '\n')
            else:
                f.write(osp.join('images/val/', img_fn))

    with open(test_fn, 'w') as f:
        for i, img_fn in enumerate(img_list_test):
            if i < len(img_list_test)-1:
                f.write(osp.join('images/test/', img_fn) + '\n')
            else:
                f.write(osp.join('images/test/', img_fn))
    
    trian_label_dir = osp.join(root_dir, 'labels/train/')
    val_label_dir = osp.join(root_dir, 'labels/val/')

    train_label_fn =osp.join(root_dir, 'train_labels.txt')
    val_label_fn =osp.join(root_dir, 'val_labels.txt')

    label_list_train = sorted(os.listdir(trian_label_dir))
    label_list_val = sorted(os.listdir(val_label_dir))

    with open(train_label_fn, 'w') as f:
        for i, img_fn in enumerate(label_list_train):
            if i < len(label_list_train)-1:
                f.write(osp.join('labels/train/', img_fn) + '\n')
            else:
                f.write(osp.join('labels/train/', img_fn))

    with open(val_label_fn, 'w') as f:
        for i, img_fn in enumerate(label_list_val):
            if i < len(label_list_val)-1:
                f.write(osp.join('labels/val/', img_fn) + '\n')
            else:
                f.write(osp.join('labels/val/', img_fn))
