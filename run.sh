export PYTHONPATH=.

DEVICE=2;
DIR_NAME="/workspace/checkpoints/";
MODEL_NAME="dp_w_adv";
HPARAMS=""

CONFIG="egs/datasets/audio/vctk/dp.yaml";
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG --exp_name $MODEL_NAME --reset --hparams=$HPARAMS
CUDA_VISIBLE_DEVICES=$DEVICE python extract_lpv.py --config $CONFIG --exp_name $MODEL_NAME

MODEL_NAME2="dp_diffgan_w_adv";
CONFIG2="egs/datasets/audio/vctk/dpg.yaml";
HPARAMS2="tts_model=$DIR_NAME$MODEL_NAME"

CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG2 --exp_name $MODEL_NAME2 --reset --hparams="$HPARAMS2"
