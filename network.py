
import dataclasses

import torch

import data

@dataclasses.dataclass(slots=True)
class Attention_param:
    hidden_size:   int
    hidden_layers: int
    head_count:    int
    head_size:     int
    dense_size:    int

    def __init__(self, config, prefix=''):
        for name in self.__annotations__:
            setattr(self, name, getattr(config, prefix+name))

        if self.head_size == -1:
            assert self.hidden_size % self.head_count == 0
            self.head_size = self.hidden_size // self.head_count
            
    def head_size_total(self):
        return self.head_count * self.head_size

class Attention(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.q_proj = torch.nn.Linear(param.hidden_size, param.head_size_total(), bias=True)
        self.k_proj = torch.nn.Linear(param.hidden_size, param.head_size_total(), bias=True)
        self.v_proj = torch.nn.Linear(param.hidden_size, param.head_size_total(), bias=True)

    def forward(self, hidden_states, timepoints=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv_shape = [batch_size, seq_len, self.param.head_count, self.param.head_size]
        # [batch_size, head_count] seq_len -> head_size
        q = self.q_proj(hidden_states).reshape(qkv_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).reshape(qkv_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).reshape(qkv_shape).transpose(1, 2)
        
        dropout_p = 0.1 if self.training else 0.0
        if timepoints is None:
            values = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            assert tuple(timepoints.shape) == (batch_size, seq_len)
            tp = timepoints[:,None,:].repeat(1, seq_len, 1)
            attn_mask = (tp - tp.mT <= 0)[:,None,:,:].expand(-1, self.param.head_count, -1, -1)
            values = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, attn_mask=attn_mask)
            
        values = values.transpose(1, 2)
        values = values.reshape(batch_size, seq_len, self.param.head_size_total())
        return values

class Decoder_layer(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.attention = Attention(param)
        self.norm1 = torch.nn.LayerNorm([param.hidden_size])
        self.norm2 = torch.nn.LayerNorm([param.hidden_size])
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(param.head_size_total(), param.dense_size, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(param.dense_size, param.hidden_size, bias=True),
            torch.nn.Dropout(p=0.1),
        )

    def forward(self, hidden_states, timepoints=None):
        # Attention
        values = self.attention(hidden_states, timepoints)
        hidden_states = self.norm1(hidden_states + values)

        # Feed-forward
        ff_output = self.dense(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states

class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        param = Attention_param(config)
        self.decoder_layers = torch.nn.ModuleList([
            Decoder_layer(param) for i in range(config.hidden_layers)
        ])
        
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        
        if config.do_rates:
            self.rate_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
            
        if config.do_packs:
            pack_param = Attention_param(config, 'pack_')
            self.pack_decoder_layers = torch.nn.ModuleList([
                Decoder_layer(pack_param) for i in range(config.pack_hidden_layers)
            ])
            self.pack_mapin = torch.nn.Linear(config.hidden_size, config.pack_hidden_size, bias=True)
            self.pack_mapout = torch.nn.Linear(config.pack_hidden_size, config.hidden_size, bias=True)

    def forward(self, input_ids, packs, token_meta=None, position_meta=None, mask_packs=True, skip_packs=False):
        hidden_states = self.embedding(input_ids)
        if token_meta is not None:
            _, m = token_meta.shape
            meta_indices = token_meta[input_ids.reshape(-1)].view(*input_ids.shape, m)
            metas = self.embedding(meta_indices)
            metas = metas.sum(2)
            hidden_states += metas
        if position_meta is not None:
            metas = self.embedding(position_meta)
            metas = metas.sum(2)
            hidden_states[:,self.config.seq_header:] += metas[:,:-1]

        if self.config.debug_images:
            import debug
        else:
            from debug import Dummy as debug
        debug.imgdump(hidden_states, 'hidden_states', scale=(-4,4))
            
        if not self.config.do_packs: skip_packs = True
        if not skip_packs:
            idx = packs.nonzero(as_tuple=True)
            if idx[0].nelement() == 0: skip_packs = True
        if not skip_packs:
            bins = torch.bincount(idx[0])
            
            col = torch.ones_like(idx[0])
            col[0] = 0
            col[1:][idx[0][:-1] != idx[0][1:]] = 1-bins[:-1]
            col.cumsum_(dim=0)

            packs_seq = torch.zeros(packs.shape[0], bins.max(), dtype=torch.long, device=packs.device)
            packs_seq[idx[0],col] = packs[idx[0],idx[1],idx[2]]

            timepoints = torch.full_like(packs_seq, packs.shape[1])
            timepoints[idx[0],col] = idx[1]
            
            debug.imgdump(bins, 'bins', force_scalar=True)
            debug.imgdump(col, 'col', force_scalar=True)
            debug.imgdump(packs_seq, 'packs_seq')
            debug.imgdump(timepoints, 'timepoints', force_scalar=True)
            
            pack_hidden_states = self.pack_mapin(self.embedding(packs_seq))
            debug.imgdump(pack_hidden_states, 'pack_hidden_states', scale=(-4,4))
            for layer in self.pack_decoder_layers:
                pack_hidden_states = layer(pack_hidden_states, timepoints)
                debug.imgdump(pack_hidden_states, 'pack_hidden_states', scale=(-4,4))

            # Make it one larger in dim 1 so that the invalid timpoints have somewhere to scatter to
            pack_output_base = torch.zeros(hidden_states.shape[0], hidden_states.shape[1]-self.config.seq_header+1, self.config.pack_hidden_size, device=packs.device)
            pack_output_base.scatter_reduce_(1, timepoints[:,:,None].expand(*pack_hidden_states.shape), pack_hidden_states, reduce='mean', include_self=False)
            pack_output = self.pack_mapout(pack_output_base[:,:-1,:])

            debug.imgdump(pack_output, 'pack_output')

        for layer_it, layer in enumerate(self.decoder_layers):
            if self.config.do_packs and not skip_packs and layer_it == self.config.pack_insert_layer:
                hidden_states[:,self.config.seq_header-1:-1,:] += pack_output
                debug.imgdump(hidden_states, f'hidden_states{layer_it}', scale=(-4,4))
            hidden_states = layer(hidden_states)
            debug.imgdump(hidden_states, f'hidden_states{layer_it}', scale=(-4,4))
            
        hidden_states = self.norm(hidden_states)

        logits_base = self.lm_head(hidden_states[:,:-1,:])
        debug.imgdump(logits_base, f'logits_base')
        if mask_packs:
            # Mask out the cards that are not in the pack
            hdr = self.config.seq_header
            logits_base[:,hdr-1:,0] = float('-inf') # Mask out token 0, since packs has 0 as fill value and I do not want the gradient of token 0 to be affected by the scatter
            logits_selected = torch.gather(logits_base[:,hdr-1:,:], 2, packs)

            logits = torch.full_like(logits_base, float('-inf'))
            logits[:,:hdr-1,:] = logits_base[:,:hdr-1,:] # Keep the header
            logits[:,hdr-1:,:].scatter_(2, packs, logits_selected)
        else:
            logits = logits_base
        
        if self.config.do_rates:
            logits_rate_base = self.rate_head(hidden_states[:,self.config.seq_header:,:])
            debug.imgdump(logits_base, f'logits_rate_base')
            card_ids = input_ids[:,self.config.seq_header:]
            n, m = card_ids.shape
            idx = torch.tril_indices(m, m, device=card_ids.device)
            indices = (idx[0] * logits_rate_base.shape[-1])[None,:] + card_ids[:,idx[1]]
            logits_rate = logits_rate_base.view(n, -1).gather(1, indices)
        else:
            logits_rate = None
        
        return logits, logits_rate
