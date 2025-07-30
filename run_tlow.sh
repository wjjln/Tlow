CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Tlow \
    --category=Sports_and_Outdoors \
    --n_codebook=128 \
    --codebook_size=256 \
    --sim_loss_weight=30 \
    --lr=0.003 \
    --temperature=0.03 \

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Tlow \
    --category=Beauty \
    --n_codebook=32 \
    --codebook_size=256 \
    --sim_loss_weight=1 \
    --lr=0.01 \
    --temperature=0.03 \

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Tlow \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=96 \
    --codebook_size=256 \
    --sim_loss_weight=10 \
    
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Tlow \
    --category=CDs_and_Vinyl \
    --n_codebook=96 \
    --codebook_size=256 \
    --lr=0.001 \
    --temperature=0.03 \
    --sim_loss_weight=3 \