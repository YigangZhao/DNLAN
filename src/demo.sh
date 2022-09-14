#!/bin/bash
#Train x2
python main.psy --epochs 1500 --model DNLAN --save DNLAN_x2 --data_test Set5 --save_dir ../output/ --dir_data ../../SrTrainingData --n_GPUs 1 --n_threads 8 --rgb_range 1 --save_models --save_results --lr 1e-4 --decay 300-600-900-1200 --chop --n_resgroups 10 --n_resblocks 4 --reduction 2 --n_hashes 3 --n_feats 128 --n_hashlength 7 --res_scale 0.1 --batch_size 16 --scale 2 --patch_size 96 --data_train DIV2K --data_range 1-800/1-5

#Test x2
python main.py --dir_data ../../SrTrainingData --save_dir ../output/ --model DNLAN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 3 --n_hashlength 7 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 128 --n_resgroups 10 --n_resblocks 4 --reduction 2 --res_scale 0.1  --pre_train model_x2.pt --test_only

