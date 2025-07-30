# Tlow
## Step 1: Obtain Text Embeddings by T5
`bash run_tlow.sh`

## Step 2: Train Tokenizer
`bash run_tokenizer.sh`

Hyper-parameters in *.gin files:

- `iterations`: # of training batches
- `input_dim`: item embedding dimension
- `n_flow` & `n_block`: model has `n_block` blocks, each block has `n_flow` flows; just use the default values
- `n_codebook`: # of codebooks, i.e., each $z$ (obtained from item embedding $x$) will be cutted as `n_codebook` parts
- `codebook_size`: size of each codebook
- `dataset_folder`: can pass multiple embedding files
- `save_dir_root`: where to save the trained Tlow models/$z$/optimizer and faiss index (PQ)

## Step 3: Train Sequential Rec Model
`bash run_tlow.sh`

## Reproduce on ``Sports`` dataset
We have provided the trained Tlow tokenizer for ``Sports`` dataset, including:
- PQ codebooks on $\mathbf{z}$ (``cache/AmazonReviews2014/Sports_and_Outdoors/processed/sentence-t5-base_Tlow_IVF1,PQ128x8.faissindex``)
- Tlow code (``cache/AmazonReviews2014/Sports_and_Outdoors/processed/sentence-t5-base_Tlow_IVF1,PQ128x8.npy``)

Train Rec model on ``Sports`` dataset:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Tlow \
    --category=Sports_and_Outdoors \
    --n_codebook=128 \
    --codebook_size=256 \
    --sim_loss_weight=30 \
    --lr=0.003 \
    --temperature=0.03 \
```

## Acknowledgement
We implement Tlow based on [RPG](https://github.com/facebookresearch/RPG_KDD2025) and [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)