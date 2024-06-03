export CUDA_VISIBLE_DEVICES=1

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate Grad-TTS-EVAL 
Converted_directory="/exp/exp4/acp23xt/Grad-TTS-Masking/eval/test/converted"
Output_directory="/exp/exp4/acp23xt/Grad-TTS-Masking/metrics/MCD/test"
Gt_directory="/exp/exp4/acp23xt/Grad-TTS-Masking/eval/test/ground_truth"

mkdir -p "$Output_directory"
sorted_folder=$(python sorted.py -d $Converted_directory)

IFS=' '

folder_list=($sorted_folder)

for folder in "${folder_list[@]}"; do
        echo "$folder"
        python MCD.py "$Gt_directory" "$folder" --outdir "$Output_directory"
        done

#python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/mean_mcd.txt -s 0 -i 100 -t MCD -x Epoch -y mean_mcd -o $Output_directory/mean_mcd.png


#python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/std_mcd.txt -s 0 -i 100 -t MCD -x Epoch -y std_mcd -o $Output_directory/std_mcd.png

