#!/bin/bash
for i in {17000..25000..1000}
do
    CUDA_VISIBLE_DEVICES=7 python sample_VAEBM.py --ebm_checkpoint ./saved_models/cifar10/dvaebm_rampup/EBM_$i.pth --savedir ./samples/iter_$i/
done
# CUDA_VISIBLE_DEVICES=7 python sample_VAEBM.py --ebm_checkpoint ./saved_models/cifar10/dvaebm_rampup/EBM_0.pth --savedir ./samples/iter_0/