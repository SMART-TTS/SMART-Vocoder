# SMART-vocoder
다양한 감정을 포함하는 한국어 DB에 대해 훈련한 SMART-Vocoder입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다.

본 모델의 특징은 다음과 같습니다.
- 여러 temporal resolution을 계층적으로 모델링할 수 있는 아키텍쳐
- 동일한 temporal resolution을 모델링하는 블록은 parameter sharing
- 효율적인 훈련을 위한 rearranged mini-batch training

다양한 감정을 포함하는 한국어 DB로 훈련하였으며, pretrained model은 [이곳](https://drive.google.com/file/d/1rqjAjRBtje5ZHhgl6OvKdvw0pGODaNrh/view?usp=sharing)에서 다운로드 가능합니다. 훈련된 모델을 사용해 LJ speech dataset을 합성한 음성 샘플은 outputs_LJ 폴더에서 확인 가능합니다.

### To-do list

- [ ] 다양한 감정을 포함하는 한국어 DB 공개 및 해당 DB에 대해 훈련할 수 있는 코드 공개


## Requirements
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

If you use LJ speech dataset, 

<pre>
<code>
python preprocess_LJ.py --in_dir LJSpeech-1.1 --out_dir datasets/preprocessed
</code>
</pre>

## Training
To train the model, run this command:
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python train.py --data_path datasets/preprocessed --bsz 8 --n_ER_blocks 4 --n_flow_blocks 5
</code>
</pre>

## Generation
To generate with the trained model, run:
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python synthesize.py --load_step 123456 --temp 0.6 --num_synth 10
</code>
</pre>

or you can run the example code with the pretrained model (pretrained/checkpoint.pth):
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python mel2audio.py
</code>
</pre>

Our pretrained model was trained on the KOR DB (not on LJ Speech).

## Results
Synthesized audio samples can be found at 'outputs_LJ/'

## Pretrained model

[Click Here](https://drive.google.com/file/d/1rqjAjRBtje5ZHhgl6OvKdvw0pGODaNrh/view?usp=sharing) to download the pretrained model on the KOR DB. 

## References
- WaveGlow: https://github.com/NVIDIA/waveglow
- FloWaveNet: https://github.com/ksw0306/FloWaveNet
