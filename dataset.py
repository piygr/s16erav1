import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path

# Hugging face datasets & tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

class BillingualDataset(Dataset):

    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.ds, self.tokenizer_src, self.tokenizer_tgt = self.get_ds()

        self.sos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.cfg['lang_src']]
        tgt_text = src_target_pair['translation'][self.cfg['lang_tgt']]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_padding_tokens_count = self.cfg['seq_len'] - len(enc_input_tokens) - 2
        dec_padding_tokens_count = self.cfg['seq_len'] - len(dec_input_tokens) - 1

        if enc_padding_tokens_count < 0 or dec_padding_tokens_count < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_padding_tokens_count, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_padding_tokens_count, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_padding_tokens_count, dtype=torch.int64)
            ],
            dim=0
        )

        
        assert encoder_input.size(0) == self.cfg['seq_len']
        assert decoder_input.size(0) == self.cfg['seq_len']
        assert label.size(0) == self.cfg['seq_len']

        return dict(
            encoder_input=encoder_input,
            encoder_str_length=len(enc_input_tokens),
            decoder_input=decoder_input,
            decoder_str_length=len(dec_input_tokens),
            label=label,
            encoder_mask=(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            decoder_mask=(decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            src_text=src_text,
            tgt_text=tgt_text
        )


    def get_all_sentences(self, lang):
        for item in self.ds:
            yield item['translation'][lang]

    def get_or_build_tokenizer(self, lang):
        tokenizer_path = Path(self.cfg['tokenizer_file'].format(lang))

        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)

            tokenizer.train_from_iterator(self.get_all_sentences(lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer

    def get_ds(self):

        ds = []
        ds_raw = load_dataset('opus_books', f"{self.cfg['lang_src']}-{self.cfg['lang_tgt']}", split='train')

        self.ds = ds_raw

        tokenizer_src = self.get_or_build_tokenizer(self.cfg['lang_src'])
        tokenizer_tgt = self.get_or_build_tokenizer(self.cfg['lang_tgt'])

        for item in ds_raw:
            src_target_pair = item
            src_text = src_target_pair['translation'][self.cfg['lang_src']]
            tgt_text = src_target_pair['translation'][self.cfg['lang_tgt']]

            enc_input_tokens = tokenizer_src.encode(src_text).ids

            if len(enc_input_tokens) > 150:
                continue

            dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids
            if len(dec_input_tokens) > len(enc_input_tokens) + 10:
                continue

            ds.append(item)

        return ds, tokenizer_src, tokenizer_tgt


def get_dataloader(cfg):

    ds = BillingualDataset(cfg)

    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size

    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])


    max_len_src = 0
    max_len_tgt = 0

    for item in ds:
        src_ids = ds.tokenizer_src.encode(item['src_text']).ids
        tgt_ids = ds.tokenizer_tgt.encode(item['tgt_text']).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Maximum length of source - {max_len_src}")
    print(f"Maximum length of target - {max_len_tgt}")


    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, ds.tokenizer_src, ds.tokenizer_tgt


def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  #returns lower triangle with diagonal as True


def collate_fn(batch):
    encoder_input_max = max(x['encoder_str_length'] for x in batch)
    decoder_input_max = max(x['decoder_str_length'] for x in batch)

    encoder_input = []
    decoder_input = []
    encoder_mask = []
    decoder_mask = []
    label = []
    src_text = []
    tgt_text = []

    for b in batch:
        encoder_input.append( b['encoder_input'][:encoder_input_max])
        decoder_input.append( b['decoder_input'][:decoder_input_max])
        encoder_mask.append( (b['encoder_mask'][0, 0, :encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0) )
        decoder_mask.append( (b['decoder_mask'][0, :decoder_input_max, :decoder_input_max]).unsqueeze(0) )
        label.append( b['label'][:decoder_input_max])
        src_text.append( b['src_text'] )
        tgt_text.append( b['tgt_text'] )

    return dict(
        encoder_input=torch.vstack(encoder_input),
        decoder_input=torch.vstack(decoder_input),
        label=torch.vstack(label),
        encoder_mask=torch.vstack(encoder_mask),
        decoder_mask=torch.vstack(decoder_mask),
        src_text=torch.vstack(src_text),
        tgt_text=torch.vstack(tgt_text)
    )