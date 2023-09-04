import torch

import torch.nn as nn

from dataset import get_dataloader

import models.TransformerV2Lightning as tl
import pytorch_lightning as pl
import config


def init(train=True):
    cfg = config.get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloader(cfg)

    '''
    src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                          tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                          N: int=6, h: int=8, d_model: int=512, d_ff: int=2048, dropout: float=0.1
    '''
    
    model = tl.build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                                 cfg['seq_len'], cfg['seq_len'], tokenizer_src, tokenizer_tgt,
                                 N=6, h=8, d_model=cfg['d_model'], d_ff=2048, dropout=0.1)


    if train:

        trainer = pl.Trainer(
            precision=16,
            max_epochs=cfg['num_epochs'],
            accelerator='gpu'
        )

        cargs = {}
        if cfg['preload']:
            cargs = dict(ckpt_path=config.get_weights_file_path(cfg, '1'))

        trainer.fit(model, train_dataloader, val_dataloader, **cargs)