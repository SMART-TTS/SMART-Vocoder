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
CUDA_VISIBLE_DEVICES=0 python train.py --bsz 8 --n_ER_blocks 4 --n_flow_blocks 5
</code>
</pre>

## Generation
To generate with the trained model, run:
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python synthesize.py --load_step 123456 --temp 0.6 --num_synth 10
</code>
</pre>

or you can run the example code:
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python synthesize.py --load_step 123456 --temp 0.6 --num_synth 10
</code>
</pre>

## Pre-trained Models
You can download pretrained models here:
* <http://example.com/>

## Results
Our model's performance is here: