import math
import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
from tokenizers import Tokenizer

from dataset import casual_mask
from config import get_config, get_weights_file_path
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x : (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)

        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)      #(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    #(1, seq_len, d_model)

        self.register_buffer('pe', pe)  # will be saved during save

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  #(batch, seq_len, d_model)

        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model is not devisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        #print('Query: ', query.shape)
        #print('Key: ', key.shape)
        #print('Value: ', value.shape)

        d_k = query.shape[-1]

        # query: (batch, h, seq_len, d_k)
        # key: (batch, h, seq_len, d_k)

        key_t = key.transpose(-2, -1) # (batch, h, d_k, seq_len)

        attention_scores = (query @ key_t) / math.sqrt(d_k) # (batch, h, seq_len, seq_len)
        if mask is not None:
            # setting -inf to mask values = 0
            #print('Attention Scores: ', attention_scores.shape)
            #print('Mask: ', mask.shape)

            attention_scores.masked_fill_(mask == 0, -1e4)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        #value: (batch, h, seq_len, d_k)
        #attenion_scores @ value : (batch, h, seq_len, d_k)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # attention

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #x : (batch, h, seq_len, d_k)
        # combine heads together

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])


    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda t: self.self_attention_block(t, t, t, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda t: self.self_attention_block(t, t, t, tgt_mask))

        # query is from decoder & key, value from encoder_output
        x = self.residual_connections[1](x, lambda t: self.cross_attention_block(t, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        #print('Decoder block: ', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            #print('Decoder layer: ', x.shape)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #x : (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        x = torch.log_softmax(self.proj(x), dim = -1)
        #print('Project layer: ', x.shape)
        return x


class TransformerV2LightningModel(pl.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer,
               src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
               src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
               tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer) -> None:

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1)
        self.cfg = get_config()

        self.writer = SummaryWriter(self.cfg['experiment_name'])
        self.last_val_batch = None

        self.metric = dict(
            total_train_steps=0,
            epoch_train_loss=[],
            epoch_train_acc=[],
            epoch_train_steps=0,
            total_val_steps=0,
            epoch_val_loss=[],
            epoch_val_acc=[],
            epoch_val_steps=0,
            train_loss=[],
            val_loss=[],
            train_acc=[],
            val_acc=[]
        )

    def encode(self, src, src_mask):
        src = self.src_embed(src) # (batch, seq_len) -> (batch, seq_len, d_model)
        src = self.src_pos(src)   # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        return self.encoder(src, src_mask) # (batch, seq_len, d_model)


    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)   # (batch, seq_len) -> (batch, seq_len, d_model)
        tgt = self.tgt_pos(tgt)     # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) # (batch, seq_len, d_model)

    def forward(self, x): #project(self, x)
        return self.projection_layer(x) #(batch, seq_len, d_model) -> (batch, seq_len, vocab_size)

    def training_step(self, train_batch, batch_idx):
        #print('--TRAIN STEP--')
        encoder_input = train_batch['encoder_input']
        decoder_input = train_batch['decoder_input']
        encoder_mask = train_batch['encoder_mask']
        decoder_mask = train_batch['decoder_mask']

        encoder_output = self.encode(encoder_input, encoder_mask)   # (batch, seq_len, d_model)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)
        proj_output = self.forward(decoder_output)  # (batch, seq_len, vocab_size)

        label = train_batch['label']

        loss = self.loss_fn( proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.log_dict({'train_loss': loss.item()})

        self.metric['total_train_steps'] += 1
        self.metric['epoch_train_steps'] += 1
        self.metric['epoch_train_loss'].append(loss.item())

        #run_validation(self, self.last_val_batch, self.tokenizer_src, self.tokenizer_tgt, self.cfg['seq_len'])
        '''
        batch_count = self.trainer.num_val_batches[0]
        print(f'Val Batch Count: {batch_count}')
        print(random.randint(0, batch_count-1), batch_idx)
        '''


        return loss

    def validation_step(self, val_batch, batch_idx):
        #print('--VAL STEP--')
        encoder_input = val_batch['encoder_input']
        decoder_input = val_batch['decoder_input']
        encoder_mask = val_batch['encoder_mask']
        decoder_mask = val_batch['decoder_mask']

        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        proj_output = self.forward(decoder_output)

        label = val_batch['label']

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.log_dict({'val_loss': loss.item()})

        self.metric['total_val_steps'] += 1
        self.metric['epoch_val_steps'] += 1
        self.metric['epoch_val_loss'].append(loss.item())

        batch_count = self.trainer.num_val_batches[0]
        if random.randint(1, batch_count) > batch_idx:
            self.last_val_batch = val_batch


    def on_validation_epoch_end(self):
        if self.metric['epoch_train_steps'] > 0:
            print('Epoch ', self.current_epoch)

            epoch_loss = 0
            for i in range(self.metric['epoch_train_steps']):
                epoch_loss += self.metric['epoch_train_loss'][i]

            epoch_loss = epoch_loss / self.metric['epoch_train_steps']
            print(f"Train Loss: {epoch_loss:5f}")
            self.metric['train_loss'].append(epoch_loss)

            self.metric['epoch_train_steps'] = 0
            self.metric['epoch_train_loss'] = []

            epoch_loss = 0
            for i in range(self.metric['epoch_val_steps']):
                epoch_loss += self.metric['epoch_val_loss'][i]

            epoch_loss = epoch_loss / self.metric['epoch_val_steps']
            print(f"Validation Loss: {epoch_loss:5f}")
            self.metric['val_loss'].append(epoch_loss)

            self.metric['epoch_val_steps'] = 0
            self.metric['epoch_val_loss'] = []
            print('------')

            run_validation(self, self.last_val_batch, self.tokenizer_src, self.tokenizer_tgt, self.cfg['seq_len'])

            print('--------------------')
            self.trainer.save_checkpoint( get_weights_file_path(self.cfg, f"{self.current_epoch:02d}") )



    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg['max_lr'], eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.cfg['max_lr'],
                                                        epochs=self.trainer.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        pct_start=0.1,
                                                        div_factor=10,
                                                        final_div_factor=10,
                                                        three_phase=True,
                                                        anneal_strategy='linear',
                                                        verbose=False
                                                        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }




def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                      N: int=6, h: int=8, d_model: int=512, d_ff: int=1024, dropout: float=0.1):

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []

    for _ in range(N // 2):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        encoder_blocks.append(
            EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        )

    decoder_blocks = []

    for _ in range(N // 2):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        decoder_blocks.append(
            DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        )


    encoder_blocks_N = encoder_blocks + [ encoder_blocks[i-1] for i in range(len(encoder_blocks), 0, -1) ]
    decoder_blocks_N = decoder_blocks + [ decoder_blocks[i-1] for i in range(len(decoder_blocks), 0, -1) ]

    encoder = Encoder(nn.ModuleList(encoder_blocks_N))
    decoder = Decoder(nn.ModuleList(decoder_blocks_N))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)


    transformer = TransformerV2LightningModel(encoder, decoder, projection_layer,
                                              src_embed, tgt_embed, src_pos, tgt_pos,
                                              tokenizer_src, tokenizer_tgt)


    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len):
    #print('--Greed Decode--')
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(src, src_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(src_mask)

        out = model.decode(decoder_input, encoder_output, src_mask, decoder_mask)
        #print('out: ', out.shape)
        prob = model.forward(out[:, -1])
        #print('prob: ', prob.shape)
        _, next_word = torch.max(prob, dim=1)
        #print('next_word: ', next_word.shape)
        #print('word: ', next_word)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item())],
            dim=1
        )

        #print('decoder_input.shape: ', decoder_input.shape)
        if next_word == eos_idx:
            break

    #print('decoder_input: ', decoder_input.squeeze(0))

    return decoder_input.squeeze(0)


def run_validation(model, data, tokenizer_src, tokenizer_tgt, max_len):
    model.eval()

    src = data['encoder_input']
    src_mask = data['encoder_mask']

    model_out = greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len)

    source_text = data['src_text'][0]
    target_text = data['tgt_text'][0]
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    expected = [target_text]
    predicted = [model_out_text]

    print(f"SOURCE := [{source_text}]")
    print(f"EXPECTED := {expected}")
    print(f"PREDICTED := {predicted}")

    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)

    print(f"Validation cer: {cer}")

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)

    print(f"Validation wer: {wer}")

    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)

    print(f"Validation BLEU: {bleu}")

    model.train()

