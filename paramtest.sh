#!/bin/bash
python train.py \
    def=lawnmower_mini \
    method=recurrent_ppo \
    method.gamma=0.999 \
    method.gae_lambda=0.98 \
    method.ent_coef=0.01 \
    # method.recurrent_preset=full_episode
    # def=lawnmower_walls_scalability \
