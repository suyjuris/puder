
import dataclasses
import datetime
import http.server
import email.utils
import re
import json
from typing import Optional
import os
import base64
import sys
import urllib.parse
import copy
import inspect
import shutil
import lzma
import pathlib
import time

import numpy as np
import lxml.etree as ET
import lxml.html
import requests
import pydantic

import configuration
import data as draftdata
import runner

def _download_file(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

@dataclasses.dataclass(slots=True)
class Draft:
    n_players: int
    n_packsize: int
    n_tokens: int
    n_rounds: int = 3
    n_header: int = 8
    
    headers: np.array = None  # player, index -> token
    packs: np.array = None  # round, player, index -> card
    picks: np.array = None  # round, player, pick -> index
    logits: np.array = None # round, player, pick, token -> value

    def init_arrays(self):
        self.headers = np.zeros([self.n_players, self.n_header], dtype=np.int16)
        self.packs = np.zeros([self.n_rounds, self.n_players, self.n_packsize], dtype=np.int16)
        self.picks = np.zeros([self.n_rounds, self.n_players, self.n_packsize], dtype=np.int8)
        self.logits = np.zeros([self.n_rounds, self.n_players, self.n_packsize, self.n_tokens], dtype=np.float32)

@dataclasses.dataclass(slots=True)
class Draft_view:
    n_packsize: int
    n_tokens: int
    n_rounds: int = 3
    n_header: int = 8
    
    headers: np.array       = None # index -> token
    packs: np.array         = None # round, pick, index -> card
    pick_cards: np.array    = None # round, pick -> card
    logits: np.array        = None # round, pick, token -> value
    logits_header: np.array = None # index, token -> value
    logits_rate: np.array  = None # round, pick, index -> value
    stored_logits: np.array = None # round, pick, token -> value (stored weights)
    stored_logits_rate: np.array = None # round, pick, index -> value (stored weights)

    def init_arrays(self):
        if self.headers is None: self.headers = np.zeros([self.n_header], dtype=np.int16)
        if self.packs is None: self.packs = np.zeros([self.n_rounds, self.n_packsize, self.n_packsize], dtype=np.int16)
        if self.pick_cards is None: self.pick_cards = np.zeros([self.n_rounds, self.n_packsize], dtype=np.int16)
        if self.logits is None: self.logits = np.zeros([self.n_rounds, self.n_packsize, self.n_tokens], dtype=np.float32)
        if self.logits_header is None: self.logits_header = np.zeros([self.n_header-1, self.n_tokens], dtype=np.float32)
        if self.logits_rate is None: self.logits_rate = np.zeros([self.n_rounds, self.n_packsize, self.n_rounds*self.n_packsize], dtype=np.float32)
        if self.stored_logits is None: self.stored_logits = np.array([])
        if self.stored_logits_rate is None: self.stored_logits_rate = np.array([])

def draft_view(draft: Draft, player: int):
    packs = np.zeros([draft.n_rounds, draft.n_packsize, draft.n_packsize], dtype=np.int16)
    for rnd in range(draft.n_rounds):
        for pick in range(draft.n_packsize):
            if rnd%2: pp = (player + pick) % draft.n_players
            else:     pp = (player - pick) % draft.n_players
            pack_orig = draft.packs[rnd,pp,:].copy()
            for i in range(pick):
                pack_orig[draft.picks[rnd,(pp+i)%draft.n_players,i]] = 0
            packs[rnd,pick,:] = pack_orig

    # This is not what we want, but something like this should exist...
    #picks = packs[draft.picks[:,players,:]],
    pick_cards = np.zeros([draft.n_rounds, draft.n_packsize], dtype=np.int16)
    for rnd in range(draft.n_rounds):
        for pick in range(draft.n_packsize):
            pick_cards[rnd,pick] = draft.packs[rnd,player,draft.picks[rnd,player,pick]]

    assert False # handle logits_header and logits_rate
    return Draft_view(
        n_packsize=draft.n_packsize,
        n_tokens=draft.n_tokens,
        n_rounds=draft.n_rounds,
        headers=draft.headers[player,:],
        packs=packs,
        pick_cards=pick_cards,
        logits=draft.logits[:,player,:,:],
    )

@dataclasses.dataclass
class Renderer_state:
    view: Draft_view
    tokens: list[str]
    token_is_card: np.array
    header_tokens: list[list[int]]
    scryfall_hotlink: bool
    index_to_url: dict[int,str]

    def _card_image_src(self, i):
        if self.scryfall_hotlink:
            src = self.index_to_url[i]
        else:
            src = f'./cards/{i}.jpg'
        return src
    
    def render_card(self, parent, i, picked=False, logit=None, rnd=None, pick_it=None, stored_logit=None):
        card_container = ET.SubElement(parent, 'div', {'class': 'card-container'})
        card_class = 'card-img' + (' picked-card' if picked else '')
        
        card = ET.SubElement(card_container, 'img', {
            'class': card_class,
            'src': self._card_image_src(i),
            'onclick': f'pickCard({rnd},{pick_it},{i})',
            'loading': 'lazy',
        })
        if logit is not None:
            weight_label = ET.SubElement(card_container, 'div', {'class': 'card-weight'})
            if stored_logit is not None:
                diff = logit - stored_logit
                weight_label.text = f'{logit:.2f} ({diff:+.2f})'
            else:
                weight_label.text = f'{logit:.2f}'
        
        if rnd is not None and pick_it is not None:
            remove_btn = ET.SubElement(card_container, 'div', {
                'class': 'card-remove-btn',
                'onclick': f'removeCard(event,{rnd},{pick_it},{i})',
            })
            remove_btn.text = '\u00d7'
    
    def _title_with_help(self, el, title, helptext):
        title_container = ET.SubElement(el, 'div', {'class': 'title-container'})
        ET.SubElement(title_container, 'h3').text = title

        # Help element
        ET.SubElement(title_container, 'div', {
            'class': 'help-icon',
            'title': helptext
        }).text = '?'
        
    def render_pick(self, parent, rnd, pick_it, pick_index, active=False):
        pick_content = ET.SubElement(parent, 'div', {
            'class': 'pick-content' + ['', ' active'][active], 
            'id': f'pick-content-{pick_index}'
        })
        
        # Left section for pack cards
        pack_section = ET.SubElement(pick_content, 'div', {'class': 'pack-section'})
        
        # Pack cards
        self._title_with_help(pack_section, f'Pack {rnd+1}, Pick {pick_it+1}', 'Numbers indicate the AI\'s estimate of how likely the card is to be picked. Click on a card to pick it.');
        
        cards_div = ET.SubElement(pack_section, 'div', {'class': 'cards'})
        pack = self.view.packs[rnd,pick_it,:]
        pick = self.view.pick_cards[rnd,pick_it]
        
        # Create a list of (card_id, weight) tuples for non-zero cards
        cards_with_weights = [(i, self.view.logits[rnd,pick_it,i]) for i in pack[pack != 0]]
        # Sort by weight in descending order
        cards_with_weights.sort(key=lambda x: x[1], reverse=True)

        # Check if we need to add an "add card" element
        # The expected number of cards is packsize - pick_it (since we've picked pick_it cards already)
        expected_cards = self.view.n_packsize - pick_it
        actual_cards = sum(pack != 0)
        
        # Render cards in sorted order
        for i, weight in cards_with_weights:
            stored_weight = self.view.stored_logits[rnd,pick_it,i] if self.view.stored_logits.size else None
            self.render_card(cards_div, i, i==pick, weight, rnd, pick_it, stored_weight)
        
        if actual_cards < expected_cards:
            # Create view_id from the URL (this is a bit of a hack, but works for the 'test' view_id)
            view_id = 'test'
            
            # Add the "add card" UI element
            add_card_container = ET.SubElement(cards_div, 'div', {
                'class': 'add-card-container',
                'onclick': f'openCardAddModal({rnd},{pick_it})'
            })
            
            add_card_box = ET.SubElement(add_card_container, 'div', {'class': 'add-card-box'})
            plus_icon = ET.SubElement(add_card_box, 'div', {'class': 'plus-icon'})
            plus_icon.text = '+'
            
            add_text = ET.SubElement(add_card_box, 'div', {'class': 'add-text'})
            add_text.text = 'Add Cards to Pack'
        
        # Previously picked cards
        rp = rnd*self.view.n_packsize + pick_it
        if rp > 0:
            self._title_with_help(pack_section, 'Previously Picked Cards', 'Numbers indicate how likely the card is to be maindecked (0 means 50%, 3 means 95%).')

            picked = ET.SubElement(pack_section, 'div', {'class': 'picked'})

            prev = list(enumerate(self.view.pick_cards.reshape(-1)[:rp]))
            prev.sort(key=lambda i: -self.view.logits_rate[rnd,pick_it,i[0]])
            for it, i in prev:
                if i == 0: continue
                stored_rate = self.view.stored_logits_rate[rnd,pick_it,it] if self.view.stored_logits_rate.size else None
                self.render_card(picked, i, logit=self.view.logits_rate[rnd,pick_it,it], stored_logit=stored_rate)
            
        # Right section for table of predictions
        info_section = ET.SubElement(pick_content, 'div', {'class': 'info-section'})

        self._title_with_help(info_section, 'Unfiltered Ratings', 'How likely a card is to be picked, but not constrained to cards from the current pack (only the set).')
        
        table = ET.SubElement(info_section, 'table', {'class': 'table'})
        thead = ET.SubElement(table, 'thead')
        thead_tr = ET.SubElement(thead, 'tr')
        ET.SubElement(thead_tr, 'th').text = 'Rating'
        ET.SubElement(thead_tr, 'th').text = 'Card Name'
        tbody = ET.SubElement(table, 'tbody')

        logits = self.view.logits[rnd,pick_it,:]
        index_map = np.where(self.token_is_card)[0]
        logits_cards = logits[self.token_is_card]
        logits_indices = index_map[np.argsort(logits_cards)]
        to_print = np.concatenate([logits_indices[:10], logits_indices[-20:]])
        for it in to_print[::-1]:
            name = self.tokens[it]
            tr = ET.SubElement(tbody, 'tr')
            ET.SubElement(tr, 'td').text = '{:.2f}'.format(logits[it])

            src = self._card_image_src(it)
            jsrc = json.dumps(src)
            
            # Create card name cell with hover preview functionality
            card_cell = ET.SubElement(tr, 'td', {
                'class': 'card-name',
                'data-card-id': str(it),
                'onmouseover': f'showCardPreview(event, {jsrc})',
                'onmouseout': 'hideCardPreview()'
            })
            card_cell.text = name

    def render(self, parent, active_pack=None, active_pick=None):
        picks = np.sum(self.view.pick_cards != 0)

        packs_drafted = picks // self.view.n_packsize
        total_rounds = min(packs_drafted + 1, 3)
        show_deckbuilding = packs_drafted == 3
        
        # Use provided parameters or fall back to heuristic
        if active_pack is not None:
            if active_pack == 3 or active_pick is None:
                active_pick = 0

            active_idx = (active_pack, min(picks, active_pack * self.view.n_packsize + active_pick))
        else:
            active_idx = (0,0)
        
        # Display meta information (header tokens)
        meta_div = ET.SubElement(parent, 'div', {'class': 'meta-info'})
        
        # Create header row with title and store weights button
        header_row = ET.SubElement(meta_div, 'div', {'class': 'meta-header-row'})
        
        meta_title = ET.SubElement(header_row, 'h3')
        meta_title.text = "Draft Metadata"

        meta_table = ET.SubElement(header_row, 'div', {'class': 'meta-table'})
        
        for idx, token_id in enumerate(self.view.headers):
            header_value = ET.SubElement(meta_table, 'div', {'class': 'meta-value'})
            select = ET.SubElement(header_value, 'select', {
                'class': 'header-select',
                'data-index': str(idx),
                'onchange': f'updateHeader(this, {idx})',
                'autocomplete': 'off',
            })

            lst = self.header_tokens[idx].copy()
            if idx:
                lst.sort(key = lambda x: -self.view.logits_header[idx-1,x])
            for token_idx in lst:
                token_name = self.tokens[token_idx][1:]
                option_attrs = {'value': str(token_idx)}
                
                if token_idx == token_id:
                    option_attrs['selected'] = ''
                
                # Create the option element
                option = ET.SubElement(select, 'option', option_attrs)
                
                # Set the clean name as a data attribute for js to use
                option_attrs['data-name'] = token_name
                
                # Format the text with the weight if applicable
                if idx:
                    w = self.view.logits_header[idx-1,token_idx]
                    option.text = f"{token_name} ({w:.2f})"
                else:
                    option.text = token_name

            div = ET.SubElement(header_value, 'div', { 'class': 'select-display-wrapper' })
            span = ET.SubElement(div, 'span', { 'class': 'select-display-text' })
            span.text = self.tokens[token_id][1:]
        
        # Add store weights button to header row
        store_weights_div = ET.SubElement(meta_div, 'div', {'class': 'store-weights-section'})
        store_weights_btn = ET.SubElement(store_weights_div, 'button', {
            'class': 'store-weights-btn',
            'onclick': 'storeWeights()',
        })
        if self.view.stored_logits.size:
            store_weights_btn.text = 'Update Stored Weights'
        else:
            store_weights_btn.text = 'Store Current Weights'
        
        # Add clear weights button (only show if stored weights exist)
        if self.view.stored_logits.size:
            clear_weights_btn = ET.SubElement(store_weights_div, 'button', {
                'class': 'clear-weights-btn',
                'onclick': 'clearWeights()',
            })
            clear_weights_btn.text = 'Clear Stored Weights'
        
        # Create container for all picks
        container = ET.SubElement(parent, 'div', {'class': 'pick-container'})
        
        # Create pack tabs container
        pack_tabs_container = ET.SubElement(container, 'div', {'class': 'pack-tabs-container'})
        
        # Create tabs for each pack
        for rnd in range(total_rounds):
            pack_tab = ET.SubElement(pack_tabs_container, 'div', {
                'class': 'pack-tab' + ['', ' active'][rnd == active_idx[0]],
                'id': f'pack-tab-{rnd}',
                'onclick': f'showPack({rnd})'
            })
            pack_tab.text = f"Pack {rnd+1}"
        
        # Add deckbuilding tab
        if show_deckbuilding:
            deckbuilding_tab = ET.SubElement(pack_tabs_container, 'div', {
                'class': 'pack-tab' + ['', ' active'][active_idx[0] == 3],
                'id': 'pack-tab-3',
                'onclick': 'showPack(3)'
            })
            deckbuilding_tab.text = "Deckbuilding"
        
        # Create pick containers for each pack
        for rnd in range(total_rounds):
            pack_content = ET.SubElement(container, 'div', {
                'class': 'pack-content' + ['', ' active'][rnd == active_idx[0]],
                'id': f'pack-content-{rnd}'
            })
            
            # Create pick tabs container for this pack
            pick_tabs_container = ET.SubElement(pack_content, 'div', {'class': 'pick-tabs-container'})
            
            # Create tabs for each pick in this pack
            start_idx = rnd * self.view.n_packsize
            end_idx = min(start_idx + self.view.n_packsize, picks+1)
            
            for i in range(start_idx, end_idx):
                active = ' active' if (rnd, i) == active_idx else ''
                pick_it = i % self.view.n_packsize
                tab = ET.SubElement(pick_tabs_container, 'div', {
                    'class': 'pick-tab' + active,
                    'id': f'pick-tab-{i}',
                    'onclick': f'showPick({i}, {rnd})'
                })
                tab.text = f"Pick {pick_it+1}"
            
            # Create pick contents for this pack
            for i in range(start_idx, end_idx):
                pick_it = i % self.view.n_packsize
                self.render_pick(pack_content, rnd, pick_it, i, active=(rnd,i) == active_idx)

        # Add deckbuilding pack content
        if show_deckbuilding:
            deckbuilding_content = ET.SubElement(container, 'div', {
                'class': 'pack-content' + ['', ' active'][active_idx[0] == 3],
                'id': 'pack-content-3'
            })

            # Deckbuilding title
            self._title_with_help(deckbuilding_content, 'Draft Deck', 'Numbers indicate how likely the card is to be maindecked (0 means 50%, 3 means 95%).')

            # Display all picked cards
            picked_cards_div = ET.SubElement(deckbuilding_content, 'div', {'class': 'picked'})

            # Get all picked cards and sort by rating
            all_picks = []
            for rnd in range(self.view.n_rounds):
                for pick in range(self.view.n_packsize):
                    card = self.view.pick_cards[rnd, pick]
                    if card != 0:
                        pick_index = rnd * self.view.n_packsize + pick
                        all_picks.append((pick_index, card))

            # Sort by rating (using the last round/pick's logits_rate as reference)
            if all_picks:
                last_rnd = self.view.n_rounds - 1
                last_pick = self.view.n_packsize - 1
                all_picks.sort(key=lambda x: -self.view.logits_rate[last_rnd, last_pick, x[0]])

            # Render all picked cards
            for pick_index, card in all_picks:
                rating = self.view.logits_rate[last_rnd, last_pick, pick_index] if all_picks else 0
                stored_rating = self.view.stored_logits_rate[last_rnd, last_pick, pick_index] if self.view.stored_logits_rate.size else None
                self.render_card(picked_cards_div, card, picked=False, logit=rating, stored_logit=stored_rating)

class Http_request_handler(http.server.BaseHTTPRequestHandler):
    def parse_path(self):
        m = re.match(r'^([^?#]*)(\?[^#]*)?(#.*)?$', self.path)
        assert m
        self.path_loc = m[1]
        self.path_anchor = m[3]
        self.path_params = {}

        if m[2]:
            self.path_params = dict(urllib.parse.parse_qsl(m[2][1:]))
            
    def _do_any(self, funcs, body=None):
        self.parse_path()
        
        parts = self.path_loc.split('/')
        func = None
        params = self.path_params.copy()
        for i in range(len(parts)):
            if i > 8: break
            key = '/'.join(parts[:i+1])
            if 1 < i+1 < len(parts): key += '/'
            func = funcs.get(key)
            if func:
                if 1 < i+1 < len(parts):
                    params['path'] = '/'.join(parts[i+1:])
                break
            
        if not func:
            return self.send_error(404, 'Not Found')
        
        sig = inspect.signature(func)
        
        if body is not None and self.headers['Content-Length'] is not None:
            try:
                body_str = body.read(int(self.headers['Content-Length']))
                body_data = json.loads(body_str)
                if not isinstance(body_data, dict):
                    return self.jerror('Body must be a JSON object')
                params['data'] = sig.parameters['data'].annotation(**body_data)
            except json.decoder.JSONDecodeError:
                return self.jerror('Body is not valid json')
            except pydantic.ValidationError as e:
                return self.jerror('Body failed validation: ' + str(e))

        for i in params:
            if i not in sig.parameters:
                return self.error(f'Invalid parameter {i}')
        skip = True
        for i in sig.parameters.values():
            if skip:
                skip = False
                continue
            if i.default == inspect.Parameter.empty and i.name not in params:
                return self.error(f'Missing parameter {i}')
            
        func(self, **params)
        
    def do_any(self, funcs, body=None):
        self._do_any(funcs, body)
            
    def do_GET(self):
        s = self.server.state
        self.do_any({
            '/': s.index,
            '/view': s.view,
            '/cards/': s.cards,
            '/res/': s.res,
        })
        
    def do_POST(self):
        s = self.server.state
        self.do_any({
            '/create': s.create,
            '/load_random': s.load_random,
            '/import_draft': s.import_draft,
            '/header': s.header,
            '/pick': s.pick,
            '/pack': s.pack,
            '/store_weights': s.store_weights,
        }, body=self.rfile)

    def error(self, msg):
        self.send_error(400, 'Invalid Request', msg)
    def jerror(self, msg):
        self.send_json({'result': 'error', 'message': msg}, code=400)

    def send_html(self, html):
        doctype = '<!DOCTYPE html>'
        html_string = lxml.html.tostring(html, pretty_print=True, include_meta_content_type=True, encoding='unicode', doctype=doctype)
        html_bytes = html_string.encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(html_bytes))
        self.end_headers()
        self.wfile.write(html_bytes)

    def send_json(self, data, code=200):
        json_bytes = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(json_bytes))
        self.end_headers()
        self.wfile.write(json_bytes)


def _build_index_to_url(runner):
    print('Building url cache...', end=' ', flush=True)

    path_cards = pathlib.Path('cache/oracle-cards.json.xz')
    if not path_cards.exists():
        data = requests.get('https://api.scryfall.com/bulk-data').json()['data']
        url = next(i['download_uri'] for i in data if i['type'] == 'oracle_cards')
        
        response = requests.get(url, headers={'Accept-Encoding': 'gzip'})
        response.raise_for_status()
        with lzma.open(path_cards, 'w') as f:
            f.write(response.content)
    
    with lzma.open(path_cards, 'rt') as f:
        data = json.load(f)

    index_to_url = {}
    for card in data:
        if card['layout'] == 'token': continue
        if card['layout'] == 'art_series': continue

        names = [card['name']]
        if 'card_faces' in card:
            names.append(card['card_faces'][0].get('name'))
        for i in list(names):
            names.append('A-' + i)
        if names[0] == "Sol'Kanar the Tainted":
            names.append("Sol'kanar the Tainted")

        if 'image_uris' in card:
            url = card['image_uris']['normal']
        else:
            url = card['card_faces'][0]['image_uris']['normal']

        for name in names:
            index = runner.tokens.index.get(name)
            if index is None: continue
            if index in index_to_url: continue
            index_to_url[index] = url

    for it, tok in enumerate(runner.tokens.token_list):
        if not tok or tok[0] in '$%': continue
        if it not in index_to_url:
            print('Error: could not find image link for', tok)
            assert False

    print('done.')
    return index_to_url

type uint = pydantic.NonNegativeInt

class Request_create(pydantic.BaseModel):
    headers: dict[str,str]
class Request_load_random(pydantic.BaseModel):
    pass
class Request_import_draft(pydantic.BaseModel):
    link: str
    
class Request_pack_card(pydantic.BaseModel):
    index: uint
    card: str
class Request_pack(pydantic.BaseModel):
    id: str
    rnd: uint
    pick: uint
    cards: list[Request_pack_card]

class Request_pick(pydantic.BaseModel):
    id: str
    rnd: uint
    pick: uint
    pick_card: uint
    
class Request_header(pydantic.BaseModel):
    id: str
    index: uint
    token: uint

class Request_store_weights(pydantic.BaseModel):
    id: str
    clear: bool = False

class Create_view_exception(Exception): pass

@dataclasses.dataclass(slots=True)
class Expansion:
    name: str
    name_tok: int
    packsize: int
    times: list[str]
    card_mask: np.array
    cards: list[str]

class Server_state:
    def __init__(self, config):
        self.views = {}

        self.config = config
        self.config.training = False
        self.config.checkpoint_load = True
        self.config.batch_size = 1
        self.config.testdata_size = self.config.testdata_size_max
        if self.config.testdata_only is None:
            self.config.testdata_only = os.path.exists(self.config.testdata_only_path) and os.path.exists('datasets_ext.json')

        self.runner = runner.Runner(self.config)
        self.runner.init_and_load()
        self.random_draft_gen = iter(self.runner.loader_test)

        self.token_is_card = np.zeros(len(self.runner.tokens), dtype=np.bool)
        for it, i in enumerate(self.runner.tokens.token_list):
            self.token_is_card[it] = i and i[0] not in '$%'

        self.header_tokens = [[] for i in configuration.Token_type]
        for i in range(len(self.runner.tokens)):
            tok = self.runner.tokens[i]
            typ = configuration.token_type(tok)
            if typ == configuration.Token_type.CARD: continue
            if typ == configuration.Token_type.META: continue
            if typ == configuration.Token_type.PADDING: continue
            idx = configuration.Token_type[typ.name.upper()].header_index
            self.header_tokens[idx].append(i)

        self.expansions = {}
        for i in self.runner.ds_meta.data:
            i_str = self.runner.tokens[i.expansion][1:]
            times = sorted([self.runner.tokens[t][1:] for t in i.times])
            card_mask = np.zeros(len(self.runner.tokens), dtype=np.bool)
            card_mask[i.cards] = True
            cards = [self.runner.tokens[ii] for ii in i.cards]
            self.expansions[i_str] = Expansion(i_str, i.expansion, i.set_packsize, times, card_mask, cards)
            self.expansions[i.expansion] = self.expansions[i_str]

        parser = lxml.html.HTMLParser()
        self.html_base = lxml.html.parse('res/view.html', parser=parser)
        self.html_index = lxml.html.parse('res/index.html', parser=parser)

        self.set_data = {}

        self.last_access = {}

        self.index_to_url = {}
        if config.scryfall_hotlink:
            import hashlib
            h = hashlib.sha256(json.dumps(self.runner.tokens.token_list).encode('utf-8')).hexdigest()[:8]
            path = pathlib.Path(f'cache/index_to_url_{h}.json')
            if not path.exists():
                itu = _build_index_to_url(self.runner)
                with path.open('w') as f:
                    json.dump(itu, f)
            with path.open() as f:
                self.index_to_url = {int(i):j for i,j in json.load(f).items()}
        
    def index(self, req):
        html = copy.deepcopy(self.html_index)

        head = html.find('.//head')
        script = ET.Element('script', {'id': 'pageData', 'type': 'application/json'})
        script.text = json.dumps({
            'expansions': [i for i in self.expansions if isinstance(i, str)]
        })
        index = min([it for it,i in enumerate(head) if i.tag == 'script'], default=len(head))
        head.insert(index, script)
        
        req.send_html(html)

    def _register_view(self, id, view):
        if len(self.views) >= 1000:
            lst = [(t,i) for i,t in self.last_access.items()]
            lst.sort()
            for _,i in lst[:200]:
                del self.last_access[i]
                del self.views[i]
                
        self.views[id] = view
        self.last_access[id] = time.time()
        
    def view(self, req, id, pack=None, pick=None):
        if not re.match('[-a-zA-Z0-9_=]+', id):
            return req.error(f'Invalid id {id}')
        
        view = self.views.get(id)
        if view is None:
            path = pathlib.Path('save') / (id + '.npz')
            if path.exists():
                view = Draft_view(**np.load(path))
                view.init_arrays()
                for k in view.__slots__:
                    if not k.startswith('n_'): continue
                    setattr(view, k, getattr(view, k).item())
                
                self._fill_logits(view)
                self._register_view(id, view)
            else:
                return req.error(f'View {id} does not exist')

        self.last_access[id] = time.time()
            
        # Parse and validate pack/pick parameters if provided
        active_pack = None
        active_pick = None
        if pack is not None:
            try:
                active_pack = int(pack)
                if not (0 <= active_pack <= view.n_rounds):
                    return req.error(f'Invalid pack {active_pack}, must be in [0,{view.n_rounds}]')
            except ValueError:
                return req.error(f'Invalid pack parameter: {pack}')
                
        if pick is not None:
            try:
                active_pick = int(pick)
                if not (0 <= active_pick < view.n_packsize):
                    return req.error(f'Invalid pick {active_pick}, must be in [0,{view.n_packsize})')
            except ValueError:
                return req.error(f'Invalid pick parameter: {pick}')

        card_mask = self.token_is_card
        card_list = []
        if exp := view.headers[configuration.Token_type.EXPANSION.header_index]:
            card_mask = self.expansions[exp].card_mask
            card_list = self.expansions[exp].cards
        renderer = Renderer_state(view, self.runner.tokens.token_list, card_mask, self.header_tokens, self.config.scryfall_hotlink, self.index_to_url)

        html = copy.deepcopy(self.html_base)

        head = html.find('.//head')
        script = ET.Element('script', {'id': 'pageData', 'type': 'application/json'})
        script.text = json.dumps({
            'view_id': id,
            'packs': view.packs.tolist(),
            'n_packsize': view.n_packsize,
            'cards': card_list,
        })
        index = min([it for it,i in enumerate(head) if i.tag == 'script'], default=len(head))
        head.insert(index, script)
        
        content = html.find('.//div[@id="content"]')
        renderer.render(content, active_pack, active_pick)
        req.send_html(html)

    def _download_card(self, name, image_path):
        data = requests.get('https://api.scryfall.com/cards/named', params={'exact': name}).json()
        if data['object'] == 'error':
            print(data)
            return f'Could not load card {name}'
            
        if 'image_uris' in data:
            link = data['image_uris']['normal']
        else:
            link = data['card_faces'][0]['image_uris']['normal']
        img = requests.get(link)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, 'wb') as f:
            f.write(img.content)

    def cards(self, req, path):
        if self.config.scryfall_hotlink:
            return req.error('Card image caching has been disabled in the configuration')
            
        m = re.match(r'^(\d+)\.jpg$', path)
        if not m: return req.error('Invalid card path')
        card_id = int(m[1])
        if not (0 <= card_id < len(self.runner.tokens)): return req.error(f'Out of bounds id {card_id}')

        image_path = f'cache/{card_id}.jpg'
        card_name = self.runner.tokens[card_id]
        if not os.path.exists(image_path):
            err = self._download_card(card_name, image_path)
            if err: return req.error(err)
            
        # Get file stats for Last-Modified and ETag
        stat = os.stat(image_path)
        file_size = stat.st_size
        last_modified = email.utils.formatdate(stat.st_mtime, usegmt=True)
        etag = f'"{card_id}-{stat.st_mtime}-{file_size}"'
        
        # Check if client has a valid cached version
        if_none_match = req.headers.get('If-None-Match')
        if_modified_since = req.headers.get('If-Modified-Since')
        
        if (if_none_match and if_none_match == etag) or \
           (if_modified_since and if_none_match is None and 
            email.utils.parsedate_to_datetime(if_modified_since) >= 
            email.utils.parsedate_to_datetime(last_modified)):
            # Client has current version
            req.send_response(304)  # Not Modified
            req.send_header('ETag', etag)
            req.send_header('Cache-Control', 'max-age=31536000, public')  # Cache for 1 year
            req.end_headers()
            return

        req.send_response(200)
        req.send_header('Content-Type', 'image/jpeg')
        req.send_header('Content-Length', file_size)
        req.send_header('Last-Modified', last_modified)
        req.send_header('ETag', etag)
        req.send_header('Cache-Control', 'max-age=31536000, public')  # Cache for 1 year
        req.end_headers()

        with open(image_path, 'rb') as f:
            shutil.copyfileobj(f, req.wfile)
        
    def _create_view(self, n_packsize, header_data=None):
        view_id = base64.urlsafe_b64encode(os.urandom(9)).decode('ascii')
        #view_id = 'test'
        view = Draft_view(n_packsize=n_packsize, n_tokens=len(self.runner.tokens))
        self._register_view(view_id, view)
        view.init_arrays()

        if header_data:
            meta = {i.name.lower(): i.header_index for i in configuration.Token_type if i.header_index is not None}
            for k,v in header_data.items():
                if k not in meta:
                    raise Create_view_exception(f"Invalid meta tag '{k}'")
                v = '$' + str(v)
                if v not in self.runner.tokens.index:
                    raise Create_view_exception(f"Invalid meta token '{v}'")
                view.headers[meta[k]] = self.runner.tokens.index[v]
        
        return view_id, view

    def res(self, req, path):
        m = re.match(r'^[a-zA-Z][-a-zA-Z0-9_.]*\.([a-z]+)$', path)
        if not m: return req.error('Invalid path')

        file_path = 'res/' + path
        if not os.path.exists(file_path): return req.error('Not found')

        mime = {
            'css': 'text/css',
            'js': 'text/javascript',
        }.get(m[1])
        if not mime: return req.error('Unknown type')
        
        # Get file stats for Last-Modified and ETag
        stat = os.stat(file_path)
        file_size = stat.st_size

        req.send_response(200)
        req.send_header('Content-Type', mime)
        req.send_header('Content-Length', file_size)
        req.end_headers()
        
        with open(file_path, 'rb') as f:
            shutil.copyfileobj(f, req.wfile)

    def _fill_logits(self, view):
        import torch
        n_cards = view.n_rounds * view.n_packsize
        n_seqlen = view.n_header + n_cards
        
        seq = torch.zeros([1, n_seqlen], dtype=torch.long)
        seq[0,:view.n_header] = torch.Tensor(view.headers)
        seq[0,view.n_header:] = torch.Tensor(view.pick_cards).view(-1)

        packs = torch.zeros([1, n_cards, self.config.seq_pack], dtype=torch.long)
        packs[0,:,:view.n_packsize] = torch.Tensor(view.packs).view(n_cards, view.n_packsize)
        
        with torch.no_grad():
            logits, logits_rate = self.runner.compute_logits(seq, packs, mask_packs=False)
            
        view.logits_header[:] = logits[0,:view.n_header-1].cpu()
        view.logits[:] = logits[0,view.n_header-1:].view(view.n_rounds, view.n_packsize, view.n_tokens).cpu()
        
        idx = np.tril_indices(n_cards)
        if view.logits_rate is None:
            view.logits_rate = np.zeros([view.n_rounds, view.n_packsize, view.n_rounds*view.n_packsize], dtype=np.float32)
        view.logits_rate.reshape(n_cards, n_cards)[idx] = logits_rate[0,:].cpu().numpy()

        mask = np.zeros_like(view.logits, dtype=np.bool)
        i,j = np.ogrid[:mask.shape[0],:mask.shape[1]]
        mask[i[...,None],j[...,None],view.packs] = 1
        mask[:,:,0] = 1 - mask[:,:,1:].max(2)
        
        view.logits_header -= np.mean(view.logits_header, -1)[...,None]
        view.logits -= np.mean(view.logits, -1, where=mask)[...,None]
        if view.stored_logits.size:
            view.stored_logits -= np.mean(view.stored_logits, -1, where=mask)[...,None]

    def _save(self, view_id, view):
        path = f'save/{view_id}.npz'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{i: getattr(view, i) for i in view.__slots__ if 'logits' not in i})

    def _save_and_fill_logits(self, view_id, view):
        self._save(view_id, view)
        self._fill_logits(view)
        
    def _send_redirect(self, req, view_id, pack=None, pick=None):
        loc = f'./view?id={view_id}'
        if pack is not None and pick is not None:
            loc += f'&pack={pack}&pick={pick}'
        loc += '\n'
        req.send_json({'result': 'success', 'location': loc})
        
    def _send_reload(self, req):
        req.send_json({'result': 'success', 'location': 'reload'})
    
    def create(self, req, data: Request_create):
        data = data.headers
        if 'expansion' not in data:
            return req.jerror(f"Must specify expansion")
        expansion = self.expansions[data['expansion']]
        
        default = {
            'event_type': 'PremierDraft',
            'time': expansion.times[-1],
            'league': 'mythic',
            'games': 'g500',
            'win_rate': 'w0.62',
            'wins': '+7',
            'losses': '-0',
        }

        try:
            view_id, view = self._create_view(expansion.packsize, default|data)
        except Create_view_exception as e:
            return req.jerror(e.args[0])
        
        self._save_and_fill_logits(view_id, view)
        self._send_redirect(req, view_id)

    def load_random(self, req, data: Request_load_random):
        picks, packs, _ = next(self.random_draft_gen)
        picks, packs = picks[0], packs[0]
        pad = (picks == 0).sum().item() // 3

        view_id, view = self._create_view(15-pad)
        
        view.headers[:] = picks[:self.config.seq_header]
        picks_cards = picks[self.config.seq_header:]
        view.pick_cards[:] = picks_cards[:view.n_packsize*view.n_rounds].view(view.n_rounds, view.n_packsize)
        
        for rnd in range(view.n_rounds):
            n = view.n_packsize
            view.packs[rnd,:] = packs[rnd*n:(rnd+1)*n,:view.n_packsize]

        self._save_and_fill_logits(view_id, view)
        self._send_redirect(req, view_id)

    def import_draft(self, req, data: Request_import_draft):
        link = data.link
        m = re.match(r'https://www\.17lands\.com/draft/([0-9a-f]+)$', link)
        if not m:
            return req.jerror(f'Invalid link')

        url_data = f'https://www.17lands.com/data/draft?draft_id={m[1]}'
        url_meta = f'https://www.17lands.com/data/event_metadata?draft_id={m[1]}'
        try:
            data = requests.get(url_data).json()
            meta = requests.get(url_meta).json()['metadata']
            #with open('out.json') as f:
            #    data, meta = json.load(f)
            #    meta = meta['metadata']
        except IOError as e:
            print('Error:', e)
            return req.jerror('Error while requesting data from 17lands')

        expansion_s = data['expansion']
        if expansion_s not in self.expansions:
            return req.jerror(f'Unknown expansion {expansion_s} (too new?)')
        
        expansion = self.expansions[expansion_s]
        if len(data['picks']) != expansion.packsize*3:
            return req.jerror(f'Unexpected packsize ({len(data["picks"])} != {expansion.packsize*3})')
        
        t = datetime.datetime.fromtimestamp(data['picks'][0]['pack_server_time']).strftime('%Y-%m')
        if t not in expansion.times:
            t = expansion.times[-1]
        
        header_data = {
            'expansion': expansion.name,
            'event_type': meta['format'],
            'time': t,
            'league': 'platinum',
            'games': 'g100',
            'win_rate': 'w0.56',
            'wins': f'+{meta["wins"]}',
            'losses': f'-{meta["losses"]}',
        }

        try:
            view_id, view = self._create_view(expansion.packsize, header_data)
        except Create_view_exception as e:
            return req.jerror(e.args[0])
        
        for i in data['picks']:
            pack_it, pick_it = i['pack_number'], i['pick_number']
            for it, j in enumerate(i['available']):
                card = j['name']
                if card not in self.runner.tokens.index:
                    return req.jerror(f'Invalid card name {card}')
                card_id = self.runner.tokens.index[card]
                view.packs[pack_it,pick_it,it] = card_id
            view.pick_cards[pack_it,pick_it] = self.runner.tokens.index[i['pick']['name']]
            
        self._save_and_fill_logits(view_id, view)
        self._send_redirect(req, view_id)

    def _download_set_data(self, set_code, path):
        url = f'https://mtgjson.com/api/v5/{set_code}.json.xz'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _download_file(url, path)

    def _get_set_data(self, set_code):
        if set_code not in self.set_data:
            path = f'cache/{set_code}.json.xz'
            if not os.path.exists(path):
                self._download_set_data(set_code, path)
            with lzma.open(path, 'rt') as f:
                self.set_data[set_code] = json.load(f)
        return self.set_data[set_code]
        
    def header(self, req, data: Request_header):
        view = self.views.get(data.id)
        if view is None:
            return req.jerror(f'View {data.id} does not exist')
        if data.token >= len(self.runner.tokens):
            return req.jerror(f'Invalid token {data.token}')
        if not self.runner.tokens[data.token].startswith('$'):
            return req.jerror(f'Token {data.token} is not a header token')
        try:
            view.headers[data.index] = data.token
        except IndexError:
            return req.jerror(f'Invalid index: {data.index}')

        self._save_and_fill_logits(data.id, view)
        self._send_reload(req)

    def pick(self, req, data: Request_pick):
        view = self.views.get(data.id)
        if view is None:
            return req.jerror(f'View {data.id} does not exist')
        try:
            view.pick_cards[data.rnd,data.pick] = data.pick_card
        except IndexError:
            return req.jerror(f'Invalid pack or pick: {data.rnd},{data.pick}')

        self._save_and_fill_logits(data.id, view)
        
        # Calculate next pack/pick
        next_pack = data.rnd
        next_pick = data.pick + 1
        
        # If we've finished this pack, move to next pack
        if next_pick >= view.n_packsize:
            next_pack += 1
            next_pick = 0
            
        # If we've finished all packs, go to deckbuilding
        if next_pack >= view.n_rounds:
            next_pack = view.n_rounds
            next_pick = 0
            
        self._send_redirect(req, data.id, next_pack, next_pick)
    
    def pack(self, req, data: Request_pack):
        view = self.views.get(data.id)
        if view is None:
            return req.jerror(f'View {data.id} does not exist')
        
        set_code = None
        set_code_tok = int(view.headers[configuration.Token_type['EXPANSION'].header_index])
        if set_code_tok:
            s = self.runner.tokens[set_code_tok]
            if configuration.token_type(s) == configuration.Token_type.EXPANSION:
                set_code = s[1:]
                set_data = self._get_set_data(set_code)
                cno_to_name = {i['number']: i['name'] for i in set_data['data']['cards']}

        try:
            pack = view.packs[data.rnd,data.pick,:]
        except IndexError:
            return req.jerror(f'Invalid pack or pick: {data.rnd},{data.pick}')

        for i in data.cards:
            if not i.card:
                tok = 0
            elif i.card[:3].isdigit():
                if set_code is None:
                    return req.jerror('Card specified by collector number, but no set code header set')
                card_name = cno_to_name.get(i.card.lstrip('0'))
                if card_name is None:
                    return req.jerror(f'Invalid collector number {i.card} for set {set_code}')
                tok = self.runner.tokens.index.get(card_name)
                if tok is None:
                    print(f'Error: could not find card {card_name}')
                    return req.jerror(f"Unknown card '{i.card}'")
            else:
                tok = self.runner.tokens.index.get(i.card)
                if tok is None:
                    return req.jerror(f"Unknown card '{i.card}'")

            if not (0 <= i.index < view.n_packsize):
                return req.jerror(f'Invalid index {i.index}, must be in [0,{view.n_packsize})')
            
            pack[i.index] = tok

        # Order cards
        mask = pack != 0
        n = np.sum(mask)
        pack[:n] = pack[mask]
        pack[n:] = 0

        picked = view.pick_cards[data.rnd,data.pick]
        if picked and picked not in pack:
            view.pick_cards[data.rnd,data.pick] = pack[0]
            
        self._save_and_fill_logits(data.id, view)
        self._send_reload(req)

    def store_weights(self, req, data: Request_store_weights):
        view = self.views.get(data.id)
        if view is None:
            return req.jerror(f'View {data.id} does not exist')
        
        if data.clear:
            view.stored_logits = np.array([])
            view.stored_logits_rate = np.array([])
        else:
            view.stored_logits = view.logits.copy()
            view.stored_logits_rate = view.logits_rate.copy()
        
        self._save(data.id, view)
        self._send_reload(req)
    
class Server_debug(http.server.HTTPServer):
    def handle_error(self, request, client_address):
        e = sys.exception()
        if isinstance(e, BrokenPipeError):
            pass
        else:
            raise e
            
class Server(http.server.HTTPServer):
    def handle_error(self, request, client_address):
        e = sys.exception()
        if isinstance(e, BrokenPipeError):
            pass
        else:
            super().handle_error(request, client_address)
        
def main(config):
    cls = Server
    if not config.allow_exceptions:
        cls = Server_debug
    httpd = cls((config.host, config.port), Http_request_handler)
    httpd.state = Server_state(config)
    httpd.serve_forever()
