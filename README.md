# SMART-vocoder
다양한 감정을 포함하는 한국어 DB에 대해 훈련한 SMART-Vocoder입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다.

본 모델의 특징은 다음과 같습니다.
- conditional variational inference 를 이용한 보코더 아키텍쳐
- stochastic 모델링 과정에서 다화자 데이터의 다양성을 잘 반영하는 구조
- 코어 모델은 Decoder, Posterior encoder, Flow, Mel encoder, Multi-scale discriminator, Multi-period discriminator로 구성


다양한 감정을 포함하는 한국어 DB로 훈련하였으며, pretrained model은 추후 업로드 될 예정입니다.

### To-do list

- [ ] 다양한 감정을 포함하는 한국어 DB 공개 및 해당 DB에 대해 훈련할 수 있는 코드 공개


## Requirements
librosa==0.8.0
matplotlib==3.3.1
numpy==1.18.5
scipy==1.5.2
tensorboard==2.3.0
torch==1.6.0
torchvision==0.7.0
Unidecode==1.1.1


## Prepare dataset:
The official KOR DB will be publicly available soon.


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
