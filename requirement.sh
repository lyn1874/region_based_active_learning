#!/bin/bash
cd DATA
datafolder=gland_data
if [ -d "$datafolder" ]; then
    echo "YEAH, $datafolder exist"
    echo "next download the resnet-v2-50 pretrained weight from tensorflow website..............."
else
    echo "download the GlaS dataset ...................................."
    wget https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip
    unzip warwick_qu_dataset_released_2016_07_08.zip
    mv 'Warwick QU Dataset (Released 2016_07_08)' gland_data
    rm -rf warwick_qu_dataset_released_2016_07_08.zip
    rm -rf __MACOSX
fi
cd ..
pwd
echo "download the resnet-v2-50 pretrained weight............."
pretrainfolder=pretrain_model
if [ -d "$pretrainfolder" ]; then
    echo "YEAH, folder $pretrainfolder exists"
else
    echo "create the folder to save resnet-v2-50 pretrained weight"
    mkdir $pretrainfolder
fi
resnetname=resnet_v2_50.ckpt
cd pretrain_model
if [ -f "$resnetname" ]; then
    echo "YEAH, $resnetname exists"
    echo "next prepare the dataset ...................."
else
    echo "download the resnet-v2-50 pretrained weight............."
    wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
    tar -xvf resnet_v2_50_2017_04_14.tar.gz
    rm resnet_v2_50_2017_04_14.tar.gz
    rm train.graph
    rm eval.graph    
fi
cd ..
echo "print current directory"
pwd
echo "prepare the dataset"
python3 -c 'import data_utils.glanddata as gd;gd.transfer_data_to_dict()'
python3 -c 'import data_utils.glanddata as gd;gd.transfer_data_to_dict_test()'
echo "-------------------------------"
echo "YEAH, FINISH PREPARING THE DATA"
echo "-------------------------------"
