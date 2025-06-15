
import csv
import dataclasses
import json
import os
import pathlib
import enum

import torch

import configuration

class Tokens:
    def __init__(self, token_list, tokens_meta=None):
        self.token_list = token_list
        self.index = {tok: i for i, tok in enumerate(self.token_list)}
        self.tokens_meta = tokens_meta

    @classmethod
    def create(cls, token_list, tokens_meta=None):
        if tokens_meta: token_list += tokens_meta['more_tokens']
        self = cls(token_list)
        
        if tokens_meta:
            self.tokens_meta = {}
            for tok, lst in tokens_meta['tokens_meta_str'].items():
                tok_id = self.index.get(tok)
                if tok_id is None:
                    tok_id = self.index['A-'+tok]
                self.tokens_meta[tok_id] = [self.index[i] for i in lst]
        return self

    def get_dict(self):
        return {
            'token_list': self.token_list,
            'tokens_meta': self.tokens_meta,
        }
                
    def __getitem__(self, i):
        return self.token_list[i]
    def __len__(self):
        return len(self.token_list)

@dataclasses.dataclass(slots=True)
class Dataset_meta:
    packs: pathlib.Path
    picks: pathlib.Path
    rates: pathlib.Path
    set_packsize: int
    seq_len: int
    packs_len: int
    n_drafts: int
    expansion: int|None = None
    times: list|None = None
    cards: object|None = None

@dataclasses.dataclass(slots=True)
class Datasets_meta:
    data: list[Dataset_meta]
    tokens_base: Tokens
    tokens_meta: dict

def generate_token_meta():
    with open('datasets.json') as f:
        datasets = json.load(f)
    tokens = datasets['tokens']
    
    more_tokens = []
    def tok(s):
        if s in tokens:
            return tokens[s]
        else:
            it = len(tokens)
            tokens[s] = it
            more_tokens.append(s)
            return it
        
    for i in range(3): tok(f'%pack={i}')
    for i in range(15): tok(f'%pick={i}')

    import lzma
    with lzma.open('AtomicCards.json.xz', 'rt') as f:
        card_data_base = json.load(f)['data']
        card_data = {}
        tokens_meta_str = {}
        for i in card_data_base.values():
            i = i[0]
            if 'faceName' in i:
                card_data[i['faceName']] = i
            card_data[i['name']] = i
            card_data[i['name'].lower()] = i

        for s, it in list(tokens.items()):
            if not s: continue
            if s[0] in '$%': continue
            if s[:2] == 'A-': s = s[2:]

            if s in card_data:
                card = card_data[s]
            else:
                card = card_data[s.lower()]

            lst = []
            lst.append(tok('%mv={:d}'.format(int(card['manaValue']))))
            if 'manaCost' in card:
                lst.append(tok('%mana=' + card['manaCost']))
            for typ in card['types']:
                lst.append(tok('%typ=' + typ))
            for col in card['colors']:
                lst.append(tok('%color=' + col))

            tokens_meta_str[s] = [more_tokens[i - len(tokens)] for i in lst]

    with open('tokens_meta.json', 'w') as f:
        data = {
            'tokens_meta_str': tokens_meta_str,
            'more_tokens': more_tokens,
        }
        json.dump(data, f, indent='  ')

def load_datasets_meta(config: configuration.Config_data):
    path = 'datasets.json'
    if config.testdata_only:
        path = 'datasets_ext.json'
    with open(path) as f:
        datasets = json.load(f)

    token_list = [None] * len(datasets['tokens'])
    for k,v in datasets['tokens'].items(): token_list[v] = k

    data = []
    for i in datasets['data']:
        i['packs'] = pathlib.Path(i['packs'])
        i['picks'] = pathlib.Path(i['picks'])
        if 'rates' in i:
            i['rates'] = pathlib.Path(i['rates'])
        else:
            i['rates'] = i['picks'].with_name(i['picks'].name.replace('pick', 'rate'))
        data.append(Dataset_meta(**i))

    try:
        with open('tokens_meta.json') as f:
            tokens_meta = json.load(f)
    except FileNotFoundError:
        tokens_meta = {}
        
    return Datasets_meta(data, Tokens(token_list), tokens_meta)

def load_dataset(lst: list[Dataset_meta], config: configuration.Config_data, datasets_json=None):
    import numpy as np
    import torch.utils.data as td
    
    datasets = []
    for it, i in enumerate(lst):
        picks = torch.from_numpy(np.fromfile(i.picks, dtype=np.int16)).view(i.n_drafts, i.seq_len)
        packs = torch.from_numpy(np.fromfile(i.packs, dtype=np.int16)).view(i.n_drafts, i.packs_len)
        rates = torch.from_numpy(np.fromfile(i.rates, dtype=np.uint8)).view(i.n_drafts, 3*i.set_packsize)

        expansions = picks[:,configuration.Token_type.EXPANSION.header_index]
        i.expansion = expansions[0].item()
        if not (expansions == i.expansion).all():
            print(f'Warning: dataset {i.picks} contains data from multiple expansions')

        i.times = torch.unique(picks[:,configuration.Token_type.TIME.header_index], return_inverse=False, return_counts=False).tolist()
        i.cards = torch.unique(picks[:,config.seq_header:], return_inverse=False, return_counts=False).numpy()

        if datasets_json is not None:
            datasets_json['data'][it]['expansion'] = i.expansion
            datasets_json['data'][it]['times'] = i.times
            datasets_json['data'][it]['cards'] = i.cards.tolist()
        
        pad = config.seq_len - i.seq_len
        if pad:
            picks = torch.cat((picks, torch.zeros(i.n_drafts, pad, dtype=picks.dtype)), 1)
            rates = torch.cat((rates, torch.zeros(i.n_drafts, pad, dtype=rates.dtype)), 1)

        a = torch.triu_indices(i.set_packsize, i.set_packsize)
        a[1] -= a[0]
        a = a[0] * config.seq_pack + a[1]
        off = i.set_packsize * config.seq_pack
        a = torch.cat([a, a + off, a + 2*off])
        a = a.expand(i.n_drafts, -1)

        npacks = torch.zeros(i.n_drafts, config.seq_cards * config.seq_pack, dtype=packs.dtype)
        npacks.scatter_(1, a, packs)
        packs = npacks.view(i.n_drafts, config.seq_cards, config.seq_pack)

        datasets.append(td.TensorDataset(picks, packs, rates))
        
    return td.ConcatDataset(datasets)


def download_json():
    import subprocess
    subprocess.run(['curl', '-LOJ', "https://mtgjson.com/api/v5/AtomicCards.json.xz"], check=True)

