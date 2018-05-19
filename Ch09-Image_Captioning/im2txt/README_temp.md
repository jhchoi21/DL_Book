### 트레이닝 데이터 준비하기
MS COCO 트레이닝 데이터를 준비하기 위해 다음의 과정을 수행합니다.
```shell
# 아래 명령어로 ${HOME} 경로를 확인합니다. e.g. /home/solaris
cat ${HOME}

# MS COCO 데이터셋을 다운받을 경로를 설정합니다. e.g. /home/solaris/im2txt/data/mscoco
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# im2txt/im2txt/data 폴더에서 아래 명령어를 실행해서 MSCOCO_DIR 경로에 MS COCO 데이터셋을 다운 받습니다.
chmod 700
./download_and_preprocess_mscoco.sh ${MSCOCO_DIR}
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

# 트레이닝 코드를 실행합니다.
# Build the model.
cd research/im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```
