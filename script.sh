if [ ! -n "$1" ]; then
    echo "Please, provide the config path"
    exit 1
fi
CUDA_VISIBLE_DEVICES=2 python ./main.py --cfg_path $1 --split=train \
# --distributed  --world_size 2

# CUDA_VISIBLE_DEVICES=0 python ./main.py --cfg_path $1 --split=test \
# #  --ckpt_path 