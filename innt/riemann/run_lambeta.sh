#!/bin/bash

python train.py --skip-adam --lbfgs-steps 2000 --lbfgs-batch-size 1048576 \
    --load-from-ckpt checkpoints \
    --ckpt-dir checkpoints_lambeta_01_beta25 \
    --lam-beta 0.1 \
    --beta 2.5

python train.py --skip-adam --lbfgs-steps 2000 --lbfgs-batch-size 1048576 \
    --load-from-ckpt checkpoints_lambeta_01_beta25 \
    --ckpt-dir checkpoints_lambeta_01_beta5 \
    --lam-beta 0.1 \
    --beta 5

python train.py --skip-adam --lbfgs-steps 2000 --lbfgs-batch-size 1048576 \
    --load-from-ckpt checkpoints_lambeta_01_beta5 \
    --ckpt-dir checkpoints_lambeta_01_beta10 \
    --lam-beta 0.1 \
    --beta 10
