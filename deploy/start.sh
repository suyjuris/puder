#!/bin/bash

set -eu

. .venv/bin/activate
python draftpredict.py --scryfall-hotlink webui
