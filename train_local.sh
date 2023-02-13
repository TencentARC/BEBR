#!/usr/bin/env bash
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# run command
# -----------------------------------------------------------------------------
GPUS=${GPUS:-1}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

export OMP_NUM_THREADS=10

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
train.py \
   --cfg configs/$1 \
   --batch-size 4096 \
   --tag coco_clip
