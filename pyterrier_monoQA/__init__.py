import sys
import math
import warnings
import itertools
import pyterrier as pt
import pandas as pd
from collections import defaultdict
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from pyterrier.transformer import TransformerBase
from typing import List
import re
from transformers import set_seed

class MonoQA(TransformerBase):
    def __init__(self, 
                 tok_model='9meo/monoQA',
                 model='9meo/monoQA',
                 batch_size=4,
                 text_field='text',
                 top_k = 5,
                 verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.top_k = top_k
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return f"MonoQA({self.model_name})"
    
    def run_model(self, input_ids):
        res = self.model.generate(input_ids)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)
    def qa(self, row):
        q, d, rank = row['query'], row[self.text_field], row['rank']
        if rank >= self.top_k:
            return ' '
        enc = self.tokenizer.encode_plus(f'Question Answering: {q} <extra_id_0> {d}', return_tensors='pt',  padding='longest')

        input_ids  = enc['input_ids'].to(self.device)
        set_seed(42)
        beam_outputs = self.model.generate(
            input_ids=input_ids,# attention_mask=attention_masks,
            do_sample=True,
            max_length=128,
#             top_k=120,
            top_k=60,
            top_p=0.98,
            early_stopping=True,
            num_beams=4,
            num_return_sequences=1
        )
        res = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)[0]
        return ' '.join(res.split()[1:])
    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        max_input_length = 512
        it = range(0, len(queries), self.batch_size)
        if self.verbose:
            it = pt.tqdm(it, desc='monoQA', unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Question Answering: {q} <extra_id_0> {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')

            input_ids  = enc['input_ids'].to(self.device)
            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                set_seed(42)
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        run = add_ranks(run)
        run['answer'] = run.progress_apply(self.qa, axis=1)
        run = run.sort_values(by=['qid', 'rank'], ascending=True)
        return run



