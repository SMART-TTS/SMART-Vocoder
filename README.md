# SMART-vocoder
다양한 감정을 포함하는 한국어 DB에 대해 훈련한 SMART-Vocoder입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다.

본 모델의 특징은 다음과 같습니다.
- conditional variational inference 를 이용한 보코더 아키텍쳐
- stochastic 모델링 과정에서 다화자 데이터의 다양성을 잘 반영하는 구조
- 코어 모델은 Decoder, Posterior encoder, Flow, Mel encoder, Multi-scale discriminator, Multi-period discriminator로 구성


다양한 감정을 포함하는 한국어 DB로 훈련하였으며, pretrained model은 추후 업데이트 될 예정입니다.

## Requirements
- librosa==0.8.0 
- matplotlib==3.3.1 
- numpy==1.18.5 
- scipy==1.5.2
- tensorboard==2.3.0
- torch==1.6.0
- torchvision==0.7.0
- Unidecode==1.1.1


## Prepare dataset:
The official KOR DB will be publicly available soon.


## Preprocessing
<pre>
<code>
python preprocess.py --wav_dir ./wav_dirs --filelists filelists/*.txt
</code>
</pre>


## Training
To train the model, run this command:
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python train.py --c configs/iitp_base.json -m iitp
</code>
</pre>

## Generation
See inference.ipynb


## Results
생성된 wav 파일은 'generated_files', 'generated_files_ms' 폴더에서 확인하실 수 있습니다.


## References
- ViTS: https://github.com/jaywalnut310/vits (MIT License)
- WaveGlow: https://github.com/NVIDIA/waveglow
- HiFiGAN: https://github.com/jik876/hifi-gan



본 프로젝트 관련 개선사항들에 대한 기술문서는 [여기](https://drive.google.com/file/d/13updcpsz7YFXOCrHVq6w0RtsQPoikRAp/view?usp=sharing)를 참고해 주세요.
