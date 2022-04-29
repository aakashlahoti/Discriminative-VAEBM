# Joint-Modelling VAEBM #

Joint-Modelling VAEBM trains a joint energy-based model built upon a pre-trained [NVAE](https://arxiv.org/abs/2007.03898) model and performs both image generation and image classification on the CIFAR-10 dataset.

The first step entails training a NVAE model:

## Training NVAE ##
Use the following command to train the NVAE from its original [codebase](https://github.com/NVlabs/NVAE). Please note that the checkpoint model linked in the NVAE codebase is incompatible with Stochastic gradient Langevin dynamics (SGLD). 

```
python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
      --num_channels_enc 128 --num_channels_dec 128 --epochs 400 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
      --weight_decay_norm 1e-1 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```

## Training the Joint-Modelling VAEBM ##
After completing the previous step, please pass the path of the saved checkpoint in the `--checkpoint` argument. Since the training of Joint-Modelling VAEBM eventually explodes owing to overfitting, the code saves a checkpoint every 500 epochs. 

We train Joint-Modelling VAEBM on CIFAR-10 using one RTX A6000 GPU. 
```
python train_VAEBM.py  --checkpoint ./checkpoints/cifar10/checkpoint.pt --experiment cifar10_exp1 --dataset cifar10 --im_size 32 --data ./data/cifar10 --num_steps 10 --wd 3e-5 --step_size 8e-5 --total_iter 22000 --alpha_s 0.2 --lr 4e-5 --max_p 0.6 --anneal_step 5000. --batch_size 32 --n_channel 128
```

## Sampling from Joint-Modelling VAEBM ##
To generate samples from our model post training, run the following command. It will generate 50000 test images in your given path.
```
python sample_VAEBM.py --checkpoint ./checkpoints/cifar_10/checkpoint.pt --ebm_checkpoint ./saved_models/cifar_10/cifar_exp1/EBM.pth --dataset cifar10 --im_size 32 --batch_size 50 --n_channel 128 --num_steps 16 --step_size 8e-5 
```
