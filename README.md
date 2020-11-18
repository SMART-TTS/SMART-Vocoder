# SMART-vocoder
This repository provides the official pytorch implementation of SMART-Vocoder

## Requirements
- Python 3.8.3
- PyTorch 1.6.0
- numpy
- soundfile
- librosa

## Prepare dataset:
The official KOR DB will be publicly available soon.
Instead, you can use [LJ speech dataset](https://keithito.com/LJ-Speech-Dataset/).


## Preprocessing
<pre>
<code>
python preprocess.py --in_dir DB --out_dir datasets/preprocessed
</code>
</pre>

## Training
To train the model, run this command:
<pre>
<code>
python train.py
</code>
</pre>

## Evaluation
To evaluate, run:
<pre>
<code>

</code>
</pre>

## Pre-trained Models
You can download pretrained models here:
* <http://example.com/>

## Results
Our model's performance is here: