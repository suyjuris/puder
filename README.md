
# puder — Predicting User Drafts by Empirical Learning

This is a transformer-based model to predict picks in MTG drafts. For a detailed description, please see the [article](https://nicze.de/philipp/articles/puder/).

There is also an [online version](https://nicze.de/philipp/puder/), which is the easiest way to run the model.

## Running it locally

(If you want to do training, please see below.)

You need Python (I used Python 3.12.8, other recent version should also work). You don't need a GPU (unless you want to do training). This model is tiny and can easily do inference on the CPU.

You also need the source code, model weights, and some data. The easiest way to get all of that is to download a release.

Then do something like the following:

    python -m venv .venv
    . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements
    python draftpredict.py webui
    
Or, if you prefer `uv`, instead do

    uv venv
    . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -r requirements
    python draftpredict.py webui

You should now be able to access the UI at http://localhost:31180/ .

## Training

(I have not validated the following. Beware.)

* You probably want a GPU.
* You then need a version of PyTorch with hardware acceleration. See [here](https://pytorch.org/get-started/locally/) for instructions.
* Download the [public 17Lands datasets](https://www.17lands.com/public_datasets), and convert them to tensors using `csv_to_tensor.cpp`.
* Install the other required packages in the same way as above.
* Runnning `python draftpredict.py train` should then create and train a new model.
