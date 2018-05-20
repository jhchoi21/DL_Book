### 트레이닝 데이터 준비하기
MS COCO 트레이닝 데이터를 준비하기 위해 다음의 과정을 수행합니다.
```shell
# 아래 명령어로 ${HOME} 경로를 확인합니다. e.g. /home/solaris
cat ${HOME}

# MS COCO 데이터셋을 다운받을 경로를 설정합니다. e.g. /home/solaris/im2txt/data/mscoco
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# im2txt/im2txt 폴더에서 아래 명령어를 실행해서 MSCOCO_DIR 경로에 MS COCO 데이터셋을 다운 받습니다.
sudo chmod 700 data/download_and_preprocess_mscoco.sh
./download_and_preprocess_mscoco.sh ${MSCOCO_DIR}
```

아래 메세지가 출력되면 모든 프로세스가 정상적으로 완료된 것입니다.
```shell
2016-09-01 16:47:47.296630: Finished processing all 20267 image-caption pairs in data set 'test'.
```


### Inception v3 체크포인트(Checkpoint) 다운로드
MS COCO 트레이닝 데이터를 준비하기 위해 다음의 과정을 수행합니다.
```shell
# Inception v3 체크포인트 파일을 다운로드 받을 경로를 지정합니다. e.g. /home/solaris/im2txt/data
INCEPTION_DIR="${HOME}/im2txt/data"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
```

### 모델 트레이닝
이제 데이터 준비가 모두 끝났습니다. 모델을 학습시키기 위해서 아래 과정을 수행합니다.
```shell
# MS COCO 데이터를 다운받은 경로를 지정합니다.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# Inception v3 체크포인트 파일을 다운로드 받은 경로를 지정합니다.
INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt"

# 모델의 학습결과(가중치)를 저장할 경로를 지정합니다.
MODEL_DIR="${HOME}/im2txt/model"

# im2txt 모듈을 사용하기 위해 환경변수를 설정합니다.
# {im2txt 폴더가 있는 경로}에 im2txt 폴더 경로를 지정합니다. e.g. /home/solaris/deep-learning-tensorflow-book-code/Ch09-Image_Captioning/im2txt
export PYTHONPATH="$PYTHONPATH:{im2txt폴더가 있는 경로}"

# 트레이닝 코드를 실행합니다.
python train.py \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```
