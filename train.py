import gin
import os
import torch
import numpy as np
import wandb
# import sys
# sys.path.append('./tokenizer/')
from accelerate import Accelerator
from tokenizer.dataset import ItemData
from tokenizer.utils import *
from tokenizer.Model import Tlow
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm
import faiss
import math
import pickle as pkl
import os

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gin_file', action='append', help='Gin configuration file')
    parser.add_argument('--gin_file', type=str, required=True) 

    args, unknown = parser.parse_known_args()
    
    gin.parse_config_file(args.gin_file)

@gin.configurable
def train(
    iterations=50000, # number of training batches
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder=[], # item embedding file paths
    save_dir_root="out/",
    split_batches=True,
    amp=False,
    wandb_logging=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    n_flow=8,
    n_block=4,
    input_dim=512, # item embedding dimension
    n_codebook=32, # number of codebooks
    codebook_size=256, # size of each codebook

):
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    train_dataset = ItemData(dataset_folder, input_dim)
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    index_dataset = ItemData(dataset_folder, input_dim)
    index_sampler = BatchSampler(SequentialSampler(index_dataset), batch_size, False)
    index_dataloader = DataLoader(index_dataset, sampler=index_sampler, batch_size=None, collate_fn=lambda batch: batch)

    train_dataloader = accelerator.prepare(train_dataloader)
    # TODO: Investigate bug with prepare eval_dataloader

    model = Tlow(
        llm_embedding_dim=input_dim,
        n_flow=n_flow,
        n_block=n_block,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="Tlow",
            config=params
        )

    start_iter = 0

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    with tqdm(initial=start_iter, total=start_iter+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], [], []]
        for iter in range(start_iter, start_iter+1+iterations):
            model.train()
            total_loss = 0

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                with accelerator.autocast():
                    model_output, _ = model(data)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss
            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.nll.cpu().item())
            losses[2].append(model_output.log_p.cpu().item())
            losses[3].append(model_output.log_det.cpu().item())
            for i in range(len(losses)):
                losses[i] = losses[i][-1000:]
            if iter % 100 == 0:
                # pbar.set_description(f'loss: {np.mean(losses[0]):.4f}, nll: {np.mean(losses[1]):.4f}, log_p: {np.mean(losses[2]):.4f}, log_det: {np.mean(losses[3]):.4f}, q_loss: {np.mean(losses[4]):.4f}')
                pbar.set_description(f'loss: {np.mean(losses[0]):.4f}, nll: {np.mean(losses[1]):.4f}, log_p: {np.mean(losses[2]):.4f}, log_det: {np.mean(losses[3]):.4f}')

            accelerator.wait_for_everyone()

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process and wandb_logging:
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss.cpu().item(),
                    "NLL": model_output.nll.cpu().item(),
                    "log_p": model_output.log_p.cpu().item(),
                    "log_det": model_output.log_det.cpu().item(),
                }
                    
            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    model.eval()
                    all_z = []
                    for batch in index_dataloader:
                        _, z = model(batch_to(batch, model.device))
                        all_z.append(z)
                    all_z = torch.cat(all_z, dim=0)
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "z": all_z,
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    # torch.save(state, save_dir_root + f"checkpoint_{n_flow}x{n_block}_{iter}.pt")
                
                if (iter+1) % eval_every == 0 or iter+1 == iterations:
                    model.eval()

                    res = faiss.StandardGpuResources()
                    res.setTempMemory(1024 * 1024 * 512)
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = n_codebook >= 56
                    faiss.omp_set_num_threads(32)
                    n_codebook_bits = int(math.log2(codebook_size))
                    index_factory = f'IVF1,PQ{n_codebook}x{n_codebook_bits}'
                    sent_embs = all_z.detach().cpu().numpy()
                    index = faiss.index_factory(
                        sent_embs.shape[1],
                        index_factory,
                        faiss.METRIC_INNER_PRODUCT
                    )
                    print(f'[TOKENIZER] Training index...')
                    if n_codebook <= 64:
                        index = faiss.index_cpu_to_gpu(res, 0, index, co)
                    index.train(sent_embs)
                    index.add(sent_embs)
                    if n_codebook <= 64:
                        index = faiss.index_gpu_to_cpu(index)

                    ivf_index = faiss.downcast_index(index)
                    invlists = faiss.extract_index_ivf(ivf_index).invlists
                    ls = invlists.list_size(0)
                    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
                    pq_codes = pq_codes.reshape(-1, invlists.code_size)
                    faiss_sem_ids = []
                    n_bytes = pq_codes.shape[1]
                    for u8code in pq_codes:
                        bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
                        code = []
                        for i in range(n_codebook):
                            code.append(bs.read(n_codebook_bits))
                        faiss_sem_ids.append(code)
                    corpus_ids = np.array(faiss_sem_ids)
                    lm_model = dataset_folder[0].split('/')[-1].split('.')[0]
                    np.save(save_dir_root + f'{lm_model}_Tlow_{index_factory}.npy', corpus_ids)
                    faiss.write_index(index, save_dir_root + f'{lm_model}_Tlow_{index_factory}.faissindex')
                    corpus_ids = torch.from_numpy(corpus_ids).int()

                    _, counts = torch.unique(corpus_ids, dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p*torch.log(p)).sum()
                    max_duplicates = counts.max() / corpus_ids.shape[0]

                    for cid in range(n_codebook):
                        _, c_counts = torch.unique(corpus_ids[:,cid], return_counts=True)
                        id_diversity_log[f"codebook_usage_{cid}"] = len(c_counts) / codebook_size

                    p_unique_ids = (counts == 1).sum() / len(corpus_ids)

                    id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                    id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()
                    id_diversity_log["p_unique_ids"] = p_unique_ids.cpu().item()

                
                if wandb_logging:
                    wandb.log({
                        **train_log,
                        **id_diversity_log
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()