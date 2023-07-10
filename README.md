# AIvoice-Parking-lot-Finder

* 자동차 주차장을 카메라로 촬영하고 그것을 기반으로 주차 가능 자리수를 판단해서 음성으로 안내해주는 프로젝트
* 본 프로젝트는 이미지 사용

## Requirement
* opencv
* python 3.9.13

## Clone code

* (Code clone 방법에 대해서 기술)

```shell
git clone https://github.com/zzz/yyy/xxxx
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정방법에 대해 기술)

```shell
python -m venv .venv
.venv/Scripts/activate

python -m pip install -U pip
python -m pip install wheel

python -m pip install openvino-dev
git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
cd open_model_zoo
python -m pip install -U pip
python -m pip install -r requirements.txt

cd open_model_zoo\demos\object-detection_demo
omz_downloader --name vehicle-detection-0202
omz_converter --name vehicle-detection-0202

cd open_model_zoo\demos\text-to-speech_demo
omz_downloader --name text-to-speech-en-0001-generation
omz_converter --name text-to-speech-en-0001-generation
omz_downloader --name text-to-speech-en-0001-generation
omz_converter --name text-to-speech-en-0001-generation
omz_downloader --name text-to-speech-en-0001-generation
omz_converter --name text-to-speech-en-0001-generation

```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)


## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /open_model_zoo\demos\text_to_speech_demo\python
python parking.py 
```

## Output

![test_0706_reuslt](https://github.com/Miniismini/AIvoice-Parking-lot-Finder/assets/131587074/b2f1b43d-a0f5-4f10-8e1f-793589ecda90)





## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
