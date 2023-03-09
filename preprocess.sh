export PYTHONPATH=.
DEVICE=3;
CONFIG="egs/datasets/audio/vctk/dp.yaml";
python data_gen/tts/runs/preprocess.py --config $CONFIG
python data_gen/tts/runs/train_mfa_align.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/tts/runs/binarize.py --config $CONFIG
