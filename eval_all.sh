export CUDA_VISIBLE_DEVICES=3

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate grad-tts-masking

python eval_all.py -c 'logs/2024-05-21' -i 100  -g 'eval/valid/ground_truth' -z 'eval/valid/converted' -m 'WAVPDFMEL'
