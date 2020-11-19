# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:27:21 2018
This script is for preparing the input for the active learning, we need to have training data, pool data,
validation data.
@author: s161488
"""
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

path_mom = "DATA/"  # NOTE, NEED TO BE MANUALLY DEFINED


def prepare_train_data(path, select_benign_train, select_mali_train):
    """
    Args:
        path: the path where the data is saved
        select_benign_train: a list of selected benign images
        select_mali_train: a list of selected malignant images
    Ops:
        First, the images, labels. edges, im_index, cls_index can be extracted from the np.load
        images: shape [85, im_h, im_w, 3]
        labels: shape [85, im_h, im_w, 1]
        im_index: shape [85]
        cls_index: shape [85]
        Start_Point will determine how many images are initialized as training image
    Output:
    
    X_train, Y_train
    X_pool, Y_pool
    X_val, Y_val    
    """
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    benign_index = np.where(np.array(classindex) == 1)
    mali_index = np.where(np.array(classindex) == 2)

    choose_index_tr = np.concatenate([benign_index[0][select_benign_train], mali_index[0][select_mali_train]], axis=0)
    benign_index_left = np.delete(range(np.shape(benign_index[0])[0]), select_benign_train)
    mali_index_left = np.delete(range(np.shape(mali_index[0])[0]), select_mali_train)

    choose_index_pl = np.concatenate([benign_index[0][benign_index_left[:27]], mali_index[0][mali_index_left[:38]]],
                                     axis=0)
    choose_index_val = np.concatenate([benign_index[0][benign_index_left[-5:]], mali_index[0][mali_index_left[-5:]]],
                                      axis=0)
    data_train = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_tr)
    data_pl = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_pl)
    data_val = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_val)

    return data_train, data_pl, data_val


def prepare_pool_data(path, aug=False):
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    select_benign_train = [0, 1, 2, 3, 4]
    select_mali_train = [2, 4, 5, 6, 7]

    benign_index = np.where(np.array(classindex) == 1)
    mali_index = np.where(np.array(classindex) == 2)

    benign_index_left = np.delete(range(np.shape(benign_index[0])[0]), select_benign_train)
    mali_index_left = np.delete(range(np.shape(mali_index[0])[0]), select_mali_train)

    choose_index_pl = np.concatenate([benign_index[0][benign_index_left[:27]], mali_index[0][mali_index_left[:38]]],
                                     axis=0)
    data_pl = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_pl)
    if aug is True:
        targ_height_npy = 528  # this is for padding images
        targ_width_npy = 784  # this is for padding images
        x_image_val, y_label_val, y_edge_val = padding_training_data(data_pl[0], data_pl[1],
                                                                     data_pl[2], targ_height_npy,
                                                                     targ_width_npy)
        data_pl = [x_image_val, y_label_val, y_edge_val]

    return data_pl


def prepare_skin_data(path, num_tr, combine=True):
    """
    choose_index_tr: worst 16+best 16
    or middle 32
    this num_tr should be 1/2*total_number_of_training_images_at_inital_step
    I have tried it for 32, then I am going to check 16

    """
    val_num_im = 96
    tot_numeric_index = np.arange(900)
    if combine is True:
        tr_select_numeric_index = np.concatenate([tot_numeric_index[:num_tr], tot_numeric_index[-num_tr:]], axis=0)
    else:
        tr_select_numeric_index = tot_numeric_index[340:(340 + num_tr * 2)]
    val_select_numeric_index = tot_numeric_index[500:(500 + val_num_im)]
    pool_numeric_index = np.delete(tot_numeric_index,
                                   np.concatenate([tr_select_numeric_index, val_select_numeric_index], axis=0))
    im_seg_score = np.load('/home/s161488/Exp_Stat/Skin_Lesion/init_segment_score.npy')
    sorted_index = np.argsort(im_seg_score)

    data_set = np.load(path, encoding='latin1').item()
    images = np.array(data_set['image'])
    labels = np.array(data_set['label'])
    edges = np.array(data_set['edge'])

    labels = np.expand_dims(labels, axis=-1)
    edges = np.expand_dims(edges, axis=-1)
    tr_select_image_index = np.sort(sorted_index[tr_select_numeric_index])
    val_select_image_index = np.sort(sorted_index[val_select_numeric_index])
    pl_select_image_index = np.sort(sorted_index[pool_numeric_index])

    imindex = np.arange(np.shape(images)[0])
    clsindex = np.ones(np.shape(images)[0])

    data_tr = extract_diff_data(images, labels, edges, imindex, clsindex, tr_select_image_index)
    data_pl = extract_diff_data(images, labels, edges, imindex, clsindex, pl_select_image_index)
    data_val = extract_diff_data(images, labels, edges, imindex, clsindex, val_select_image_index)

    return data_tr[:3], data_pl[:3], data_val[:3]


def prepare_test_data(path):
    if "_test" not in path:
        print("-------I am loading the data from pool set------")
        return prepare_pool_data(path)
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    return images, labels, edges, imageindex, classindex


def generate_batch(x_image_tr, y_label_tr, y_edge_tr, y_binary_mask_tr, batch_index, batch_size):
    im_group = [x_image_tr, y_label_tr, y_edge_tr, y_binary_mask_tr]
    im_batch = []
    for single_im in im_group:
        _im_batch = single_im[batch_index:(batch_size + batch_index)]
        im_batch.append(_im_batch)
    batch_index = batch_index + batch_size
    return im_batch[0], im_batch[1], im_batch[2], im_batch[3], batch_index


def padding_training_data(x_image, y_label, y_edge, target_height, target_width):
    """Each image has different size, so I need to pad it with zeros to make sure each image has the same size.
       Then I can perform random crop, rotation and other augmentation on per batch instead of per image
    """
    x_im_pad, y_la_pad, y_ed_pad = [], [], []
    num_image = np.shape(x_image)[0]
    for i in range(num_image):
        image_pad, label_pad, edge_pad = padding_zeros(x_image[i], y_label[i], y_edge[i], target_height, target_width)
        x_im_pad.append(image_pad)
        y_la_pad.append(label_pad)
        y_ed_pad.append(edge_pad)
    x_im_pad = np.reshape(x_im_pad, [num_image, target_height, target_width, 3])
    y_la_pad = np.reshape(y_la_pad, [num_image, target_height, target_width, 1])
    y_ed_pad = np.reshape(y_ed_pad, [num_image, target_height, target_width, 1])
    return x_im_pad, y_la_pad, y_ed_pad


def padding_zeros(image, label, edge, target_height, target_width):
    im_h, im_w, _ = np.shape(image)
    delta_w = target_width - im_w
    delta_h = target_height - im_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    image_pad = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant')
    label_pad = np.pad(label, ((top, bottom), (left, right)), mode='constant')
    edge_pad = np.pad(edge, ((top, bottom), (left, right)), mode='constant')
    return image_pad, label_pad, edge_pad


def extract_diff_data(image, label, edge, im_index, cls_index, choose_index):
    new_data = [[] for _ in range(5)]
    old_data = [image, label, edge, im_index, cls_index]
    for i in choose_index:
        for single_new, single_old in zip(new_data, old_data):
            single_new.append(single_old[i])
    return new_data[0], new_data[1], new_data[2], new_data[3], new_data[4]


def aug_train_data(image, label, edge, binary_mask, batch_size, aug, imshape):
    """This function is used for performing data augmentation. 
    image: placeholder. shape: [Batch_Size, im_h, im_w, 3], tf.float32
    label: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    edge: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    binary_mask: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    aug: bool
    imshape: [targ_h, targ_w, ch]
    Outputs:
    image: [Batch_Size, targ_h, targ_w, 3]
    label: [Batch_Size, targ_h, targ_w, 1]
    edge: [Batch_Size, targ_h, targ_w, 1]
    binary_mask: [Batch_Size, targ_h, targ_w, 1]
    
    """
    image = tf.cast(image, tf.int64)
    bigmatrix = tf.concat([image, label, edge, binary_mask], axis=3)
    target_height = imshape[0].astype('int32')
    target_width = imshape[1].astype('int32')
    if aug is True:
        bigmatrix_crop = tf.random_crop(bigmatrix, size=[batch_size, target_height, target_width, 6])
        bigmatrix_crop = tf.cond(tf.less_equal(tf.reduce_sum(bigmatrix_crop[:, :, :, 5]), 10),
                                 lambda: tf.image.resize_image_with_crop_or_pad(bigmatrix, target_height, target_width),
                                 lambda: bigmatrix_crop)
        # instead of judging by label, should do it by the binary mask!
        k = tf.random_uniform(shape=[batch_size], minval=0, maxval=6.5, dtype=tf.float32)
        bigmatrix_rot = tf.contrib.image.rotate(bigmatrix_crop, angles=k)
        image_aug = tf.cast(bigmatrix_rot[:, :, :, 0:3], tf.float32)
        label_aug = bigmatrix_rot[:, :, :, 3]
        edge_aug = bigmatrix_rot[:, :, :, 4]
        binary_mask_aug = bigmatrix_rot[:, :, :, 5]
    else:
        bigmatrix_rot = tf.image.resize_image_with_crop_or_pad(bigmatrix, target_height, target_width)
        image_aug = tf.cast(tf.cast(bigmatrix_rot[:, :, :, 0:3], tf.uint8), tf.float32)
        label_aug = tf.cast(bigmatrix_rot[:, :, :, 3], tf.int64)
        edge_aug = tf.cast(bigmatrix_rot[:, :, :, 4], tf.int64)
        binary_mask_aug = tf.cast(bigmatrix_rot[:, :, :, 5], tf.int64)
    return image_aug, tf.expand_dims(label_aug, -1), tf.expand_dims(edge_aug, -1), tf.expand_dims(binary_mask_aug, -1)


def collect_test_data(resize=True):
    test_a_path = path_mom + "/Data/glanddata_testa.npy"
    test_b_path = path_mom + "/Data/glanddata_testb.npy"
    image_tot, label_tot = [], []
    target_height, target_width = 528, 784
    for single_path in [test_a_path, test_b_path]:
        data_set = np.load(single_path, allow_pickle=True).item()
        images = data_set['image']
        y_label_pl = data_set['label']
        y_edge_pl = data_set['edge']
        x_image_val = []
        y_label_val = []
        if resize is True:
            for single_im, single_label in zip(images, y_label_pl):
                for _im_, _path_ in zip([single_im, single_label], [x_image_val, y_label_val]):
                    _im_ = cv2.resize(_im_, dsize=(784, 528), interpolation=cv2.INTER_CUBIC)
                    _path_.append(_im_)
        else:
            x_image_val, y_label_val, y_edge_val = padding_training_data(images, y_label_pl, y_edge_pl, target_height,
                                                                         target_width)
        image_tot.append(x_image_val)
        label_tot.append(y_label_val)
    image_tot = np.concatenate([image_tot[0], image_tot[1]], axis=0)
    label_tot = np.concatenate([label_tot[0], label_tot[1]], axis=0)
    print("The shape of the test images", np.shape(image_tot))
    return image_tot, label_tot


def save_im():
    im_tot, la_tot = collect_test_data()
    rand_value = np.random.choice(np.arange(len(im_tot)), 3, replace=False)
    for i in rand_value:
        fig = plt.figure(figsize=(10, 4))
        im_ = im_tot[i]
        la_ = la_tot[i]
        la_judge = (la_ != 0)
        for iterr, single_im in enumerate([im_, la_, la_judge]):
            ax = fig.add_subplot(1, 3, iterr + 1)
            ax.imshow(single_im)
        plt.savefig('/home/blia/im_%d.pdf' % i, pad_inches=0, box_inches='tight')
