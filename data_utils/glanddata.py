# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:43:17 2018
This script is utilized for creating the data 
    data['image'] = image [1, Number_of_Image]
    data['label'] = label [1, Number_of_Image]
    data['edge'] = Edge [1, Number_of_Image]
    data['boundingbox'] = Bounding_Box [1, Number_of_Image, Maximum_Num_of_Instances(30), 4](x_min,x_max,y_min,y_max)
@author: s161488

"""
import os
import numpy as np
import scipy
from scipy import misc
from skimage.morphology import dilation, disk
from scipy import ndimage
import matplotlib.pyplot as plt

path_mom = "DATA/"  # the directory for saving data
# path_mom = "/Users/bo/Documents/Exp_Data/"
path_use = path_mom + 'Data'
if not os.path.exists(path_use):
    os.makedirs(path_use)


def get_filename_train_list():
    path = path_mom + '/gland_data/'
    all_file = os.listdir(path)
    train_label_filename = [path + filename for filename in sorted(all_file)
                            if filename.startswith("train") & filename.endswith("_anno.bmp")]
    train_image_filename = [filename.strip().split("_anno")[0] + ".bmp" for filename in train_label_filename]
    print("Total number of training images:", np.size(train_image_filename))

    return train_image_filename, train_label_filename


def get_filename_test_list():
    path = path_mom + '/gland_data/'
    all_file = os.listdir(path)
    test_a_label_filename = [path + filename for filename in sorted(all_file)
                             if filename.startswith("testA") & filename.endswith("_anno.bmp")]
    test_a_image_filename = [filename.strip().split("_anno")[0] + ".bmp" for filename in test_a_label_filename]
    test_b_label_filename = [path + filename for filename in sorted(all_file)
                             if filename.startswith("testB") & filename.endswith("_anno.bmp")]
    test_b_image_filename = [filename.strip().split("_anno")[0] + ".bmp" for filename in test_b_label_filename]
    print("Total number of testA image:", np.size(test_a_image_filename))
    print("Total number of testB image:", np.size(test_b_image_filename))
    return test_a_image_filename, test_a_label_filename, test_b_image_filename, test_b_label_filename


def read_gland_training_data(im_list, la_list):
    """This function is utilized to read the image and label
    """
    images = []
    labels = []
    images_index = []
    index = 0

    for im_filename, la_filename in zip(im_list, la_list):
        im = scipy.misc.imread(im_filename)
        im_index = int(im_filename.strip().split("train_")[1].strip().split(".bmp")[0])
        la = scipy.misc.imread(la_filename)
        images_index.append(im_index)
        images.append(im)
        labels.append(la)
        index = index + 1

    print('%d Gland Training images are loaded' % index)
    return images, labels, images_index


def read_gland_test_data(im_list, la_list, name):
    """This function is utilized to read the image and label
    name: "testA_" or "testB_"
    """
    images = []
    labels = []
    images_index = []
    index = 0

    for im_filename, la_filename in zip(im_list, la_list):
        im = scipy.misc.imread(im_filename)
        im_index = int(im_filename.strip().split(name)[1].strip().split(".bmp")[0])
        la = scipy.misc.imread(la_filename)
        images_index.append(im_index)
        images.append(im)
        labels.append(la)
        index = index + 1

    print('%6.1f Gland test %s images are loaded' % (index, name))
    return images, labels, images_index


def extract_edge(label):
    """This function is utilized to extract the edge from the ground truth
    Args:
        label: The ground truth of all images 
        shape [Number_of_image, Image_Height, Image_Width,1]
    Returns:
        The edge feature map. If the pixel belongs to edge, then the label is set to be one. 
        If the pixel doesn't belong to edge, then the label is set to be zero.
        shape: [Number_of_image, Image_height, Image_Width,1]
    
    The requirement for this function is scipy!
    """
    selem = disk(3)
    edge_feat = []
    for la_sep in label:
        sx = ndimage.sobel(la_sep, axis=0, mode='constant')
        sy = ndimage.sobel(la_sep, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        row = (np.reshape(sob, -1) > 0) * 1
        edge_sep = np.reshape(row, np.shape(sob))
        edge_sep = dilation(edge_sep, selem)
        edge_feat.append(edge_sep.astype('int64'))
    return edge_feat


def extract_benign_malignant(test_name):
    path = path_mom + '/gland_data/Grade.csv'
    fd = open(path)
    index = []
    class_index = []
    for i in fd:
        i = i.strip().split("\r")
        for j in i:
            j = j.strip().split(",")
            train_index = [k for k in j if test_name in k]
            if train_index:
                index.append(int(train_index[0].split("_")[-1]))
                for k in j:
                    if k.endswith("ign"):
                        class_index.append(1)
                    if k.endswith("nant"):
                        class_index.append(2)
    return index, class_index


def transfer_data_to_dict():
    """This function is utilized to save the original image in a dictionary
    Return:
        data['image'] = image [1, Number_of_Image*4] 85*4
        data['label'] = label [1, Number_of_Image*4]
        data['edge'] = Edge [1, Number_of_Image*4]
    Requirements:
        from collections import defaultdict
    """
    from collections import defaultdict
    tr_im, tr_la = get_filename_train_list()
    image, label, image_index = read_gland_training_data(tr_im, tr_la)
    im_ind, class_index = extract_benign_malignant('train')
    cla_ind_fin = []
    for i in range(np.shape(image_index)[0]):
        cla_ind_fin.append(class_index[int(np.where(np.array(im_ind) == image_index[i])[0])])
    edge = extract_edge(label)

    data = defaultdict(list)
    data['image'] = image
    data['label'] = label
    data['edge'] = edge
    data['ImageIndex'] = image_index
    data['ClassIndex'] = cla_ind_fin
    filename = path_mom + "/Data/glanddata.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Oh, this is the first time of creating this file")
        print("Creating the training data npy for GlaS")

    np.save(filename.split(".")[0], data)


# The standardeivation for the boudnign box ([ 158.73026619,  241.80872181,  159.79253117,  240.67909876])
def transfer_data_to_dict_test():
    """This function is utilized to save the test image in a dictionary
    Return:
        data['image'] = image [1, Number_of_Image*4] 85*4
        data['label'] = label [1, Number_of_Image*4]
        data['edge'] = Edge [1, Number_of_Image*4]
    Requirements:
        from collections import defaultdict
    """
    from collections import defaultdict
    te_a_im, te_a_la, te_b_im, te_b_la = get_filename_test_list()
    imagea, labela, image_indexa = read_gland_test_data(te_a_im, te_a_la, "testA_")
    im_inda, class_indexa = extract_benign_malignant("testA")
    cla_ind_fina = []
    for i in range(np.shape(image_indexa)[0]):
        cla_ind_fina.append(class_indexa[int(np.where(np.array(im_inda) == image_indexa[i])[0])])
    imageb, labelb, image_indexb = read_gland_test_data(te_b_im, te_b_la, "testB_")
    im_indb, class_indexb = extract_benign_malignant("testB")
    cla_ind_finb = []
    for i in range(np.shape(image_indexb)[0]):
        cla_ind_finb.append(class_indexb[int(np.where(np.array(im_indb) == image_indexb[i])[0])])

    image_benign = []
    label_benign = []
    image_index_benign = []
    image_mali = []
    label_mali = []
    image_index_mali = []
    for index, class_single in enumerate(cla_ind_fina):
        if class_single == 1:
            image_benign.append(imagea[index])
            label_benign.append(labela[index])
            image_index_benign.append(image_indexa[index])
        else:
            image_mali.append(imagea[index])
            label_mali.append(labela[index])
            image_index_mali.append(image_indexa[index])

    for index, class_single in enumerate(cla_ind_finb):
        if class_single == 1:
            image_benign.append(imageb[index])
            label_benign.append(labelb[index])
            image_index_benign.append(image_indexb[index])
        else:
            image_mali.append(imageb[index])
            label_mali.append(labelb[index])
            image_index_mali.append(image_indexb[index])
    edge_benign = extract_edge(label_benign)
    edge_mali = extract_edge(label_mali)

    cla_ind_benign = np.repeat(1, 37)
    cla_ind_mali = np.repeat(2, 43)
    data = defaultdict(list)
    data['image'] = image_benign
    data['label'] = label_benign
    data['edge'] = edge_benign
    data['ImageIndex'] = image_index_benign
    data['ClassIndex'] = cla_ind_benign
    filename = path_mom + "/Data/glanddata_test_benign.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Creating GlaS test data (benign)")

    print("Saving the data in path:", filename.split(".")[0])
    np.save(filename.split(".")[0], data)
    data1 = defaultdict(list)
    data1['image'] = image_mali
    data1['label'] = label_mali
    data1['edge'] = edge_mali
    data1['ImageIndex'] = image_index_mali
    data1['ClassIndex'] = cla_ind_mali
    filename = path_mom + "/Data/glanddata_test_mali.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Creating GlaS test data (malignant)")
    print("Saving the data in path:", filename.split(".")[0])
    np.save(filename.split(".")[0], data1)
