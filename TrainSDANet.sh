#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=3 python -u  ./Source/main.py \
                      --gpu 1 --phase train \
                      --dataset KITTI \
                      --modelName SDANet \
                      --modelDir ./PAModel/ \
                      --auto_save_num 1 \
                      --imgNum 4400 \
                      --valImgNum 0 \
                      --maxEpochs 800 \
                      --learningRate 0.0002 \
                      --outputDir ./Result/ \
                      --trainListPath ./Dataset/trainlist_Kitti_Sceneflow.txt \
                      --trainLabelListPath ./Dataset/labellist_Kitti_Sceneflow.txt \
                      --corpedImgWidth 256 \
                      --corpedImgHeight 128 \
                      --batchSize 1 \
                      --pretrain false > Train.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TrainKitti.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
