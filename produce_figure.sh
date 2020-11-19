#!/bin/bash
expdir=Exp_Stat
if [ -d "$expdir" ]; then
    echo "folder $expdir already exists"
    echo "next step, download the statistics to reproduce figures"
else
    mkdir $expdir
    echo "make folder $expdir for saving statistics"
fi
cd $expdir
filename=calibration_score
if [ -d "$filename" ]; then
    echo "YEAH! $filename exists"
else
    echo "$filename does not exist"
    echo "Download the file..................................."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VT_6mMA9ksDV8O3shYEXEehrMPDktmVZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VT_6mMA9ksDV8O3shYEXEehrMPDktmVZ" -O calibration_score.tar.gz && rm -rf /tmp/cookies.txt
    echo "-----unziping the datafile"
    tar -xzvf calibration_score.tar.gz
    mv use_stat calibration_score
    rm calibration_score.tar.gz
fi
cd ..
echo "Start to reproduce figures"
python3 visualize_calibration_score.py --save True --path Exp_Stat/calibration_score/
echo "-----------------------------------"
echo "YEAH, FINISH REPRODUCING FIGURES :)"
echo "-----------------------------------"