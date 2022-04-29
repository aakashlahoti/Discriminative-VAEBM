# Discriminative-VAEBM #
Adding a discriminator conjoined with the VAEBM Model.

VAEBM trains an energy network to refine the data distribution learned by an [NVAE](https://arxiv.org/abs/2007.03898), where the enery network and the VAE jointly define an Energy-based model.
The NVAE is pretrained before training the energy network, and please refer to [NVAE's implementation](https://github.com/NVlabs/NVAE) for more details about constructing and training NVAE.

## Training NVAE ##
We use the following commands on each dataset for training the NVAE backbone. To train NVAEs, please use its original [codebase](https://github.com/NVlabs/NVAE) with commands given here.
#### CIFAR-10 (8x 16-GB GPUs) ####
```
python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
      --num_channels_enc 128 --num_channels_dec 128 --epochs 400 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
      --weight_decay_norm 1e-1 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```

## Training VAEBM ##
We use the following commands on each dataset for training VAEBM. Note that you need to train the NVAE on corresponding dataset before running the training command here.
After training the NVAE, pass the path of the checkpoint to the `--checkpoint` argument.

Note that the training of VAEBM will eventually explode (See Appendix E of our paper), and therefore it is important to save checkpoint regularly. After the training explodes, stop running the code and use the last few saved checkpoints for testing.
#### CIFAR-10 ####

We train VAEBM on CIFAR-10 using one 32-GB V100 GPU. 
```
python train_VAEBM.py  --checkpoint ./checkpoints/cifar10/checkpoint.pt --experiment cifar10_exp1 --dataset cifar10 --im_size 32 --data ./data/cifar10 --num_steps 10 --wd 3e-5 --step_size 8e-5 --total_iter 30000 --alpha_s 0.2 --lr 4e-5 --max_p 0.6 --anneal_step 5000. --batch_size 32 --n_channel 128
```

## Sampling from VAEBM ##
To generate samples from VAEBM after training, run ```sample_VAEBM.py```, and it will generate 50000 test images in your given path. When sampling, we typically use 
longer Langvin dynamics than training for better sample quality, see Appendix E of the [paper](https://arxiv.org/abs/2010.00654) for the step sizes and number of steps we use to obtain test samples
for each dataset. Other parameters that ensure successfully loading the VAE and energy network are the same as in the training codes. 

For example, the script used to sample CIFAR-10 is
```
python sample_VAEBM.py --checkpoint ./checkpoints/cifar_10/checkpoint.pt --ebm_checkpoint ./saved_models/cifar_10/cifar_exp1/EBM.pth --dataset cifar10 --im_size 32 --batch_size 40 --n_channel 128 --num_steps 16 --step_size 8e-5 
```
