
import enum
import dataclasses
from typing import Optional

class Token_type(enum.Enum):
    CARD       = 0
    PADDING    = 1
    EXPANSION  = 2, 0
    EVENT_TYPE = 3, 1
    TIME       = 4, 2
    LEAGUE     = 5, 3
    GAMES      = 6, 4
    WIN_RATE   = 7, 5
    WINS       = 8, 6
    LOSSES     = 9, 7
    META       = 10

    def __init__(self, index, header_index=None):
        self.index = index
        self.header_index = header_index
    
def token_type(s):
    if not s: return Token_type.PADDING
    elif s[0] == '%': return Token_type.META
    elif s[0] == '$':
        ss = s[1:]
        if ss.isupper() and ss[0].isalpha(): return Token_type.EXPANSION
        elif ss[-1].isdigit():
            if   ss[0] == 'g': return Token_type.GAMES
            elif ss[0] == 'w': return Token_type.WIN_RATE
            elif ss[0] == '+': return Token_type.WINS
            elif ss[0] == '-': return Token_type.LOSSES
            else:
                assert ss[4] == '-'
                return Token_type.TIME
        else:
            if ss[0].isupper():
                assert ss in ('PremierDraft', 'TradDraft')
                return Token_type.EVENT_TYPE
            else:
                return Token_type.LEAGUE
    else:
        return Token_type.CARD


@dataclasses.dataclass(slots=True)
class Config_data:
    seq_header: int = 8
    seq_pack:   int = 15
    seq_cards:  int = seq_pack * 3
    seq_len:    int = seq_header + seq_cards

    def merge(self, values):
        for k,v in values.items():
            setattr(self, k, v)

    @classmethod
    def keys(cls):
        for base in cls.__bases__:
            if base == object: continue
            for i in base.keys():
                yield i
        for i in cls.__slots__:
            yield i
    
@dataclasses.dataclass(slots=True)
class Config_model(Config_data):
    hidden_size:   int = 768
    hidden_layers: int = 5
    head_count:    int = 12
    head_size:     int = -1
    dense_size:    int = 1536
    vocab_size:    int = -1
    
    pack_hidden_size:   int = 256
    pack_hidden_layers: int = 2
    pack_head_count:    int = 4
    pack_head_size:     int = -1
    pack_dense_size:    int = 512
    pack_insert_layer:  int = 2

    meta_cardinfo: bool = False
    meta_pickinfo: bool = False

    do_rates: bool = True
    do_packs: bool = True

    def meta_any(self):
        return self.meta_cardinfo or self.meta_pickinfo

    def merge(self, values):
        for k,v in values.items():
            setattr(self, k, v)

@dataclasses.dataclass(slots=True)
class Config_load(Config_model):
    training: bool = False
    checkpoint_load: Optional[bool] = None # default to (not training)
    checkpoint_load_path: Optional[str] = None
    checkpoint_load_strict: bool = True

    testdata_size: int = 128
    testdata_size_max: int = 1024
    traindata_skip: int = 0
    testdata_only: bool = None
    testdata_only_path: str = 'tensors/testdata.pt'

    _args_config: Optional[dict] = None
    
    def get_checkpoint_load(self):
        if self.checkpoint_load is None:
            return not self.training
        return self.checkpoint_load
            
@dataclasses.dataclass(slots=True)
class Config_train(Config_load):
    batch_size:    int = 128
    epochs:        int = 1
    
    learning_rate: float = 1e-4
    adam_beta1:    float = 0.9
    adam_beta2:    float = 0.999
    adam_eps:      float = 1e-8
    grad_clip:     float = 1.0
    learning_rate_min: float = 1e-5
    lr_warmup:     int = 32 # 0 to disable
    schedule_free: bool = True

    rate_loss_fac:  float = 0.3
    rate_loss_posw: float = 1.25

    pack_skip_frac: float = 0.5
    
    train_duration_max: float = float('inf')
    train_instances_max: int = -1
    train_continue: bool = True
    train_cutoffs: list[(float, float)] = dataclasses.field(default_factory=list)

    filter_hq: bool = False

    device: str = ''
    silent: bool = False
    
    period_print: float = 4.0  # seconds
    period_save:  float = 32.0 # seconds
    period_test:  int   = 32   # batches, -1 to disable

    checkpoint_save: bool = True
    checkpoint_save_optim: bool = True
    
    debug_memory: bool = False
    debug_profile: bool = False
    debug_images: bool = False
    tensorboard: bool = True
    tensorboard_embedding: bool = False

@dataclasses.dataclass(slots=True)
class Config_webui(Config_train):
    host: str = ''
    port: int = 31180
    allow_exceptions: bool = True
    scryfall_hotlink: bool = False

Config = Config_webui
