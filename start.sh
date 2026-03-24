#!/bin/bash
# RiemannSD-Net training launcher script.
#
# Usage: bash start.sh
#
# Modify the variables below to configure your experiment.

# ─────────────────────────── Configuration ───────────────────────────
encoder_name="dual_ReD_ecapa_cat"  # resnet34 | ecapa_tdnn | dual_mamba_cat | dual_ecapa_cat |
                                    # dual_conformer_cat | dual_transformer_cat | dual_conv_conformer_cat |
                                    # dual_resnet34_cat | dual_aasist_cat | dual_ReD_ecapa_cat
embedding_dim=512
loss_name="ChebySDAAMSoftmax"      # amsoftmax | AAMsoftmax | ChebyAAMSoftmax |
                                    # ChebySDAAMSoftmax | RiemannSDAAMSoftmax |
                                    # SPAAMsoftmax | RiemannianTangentAAM

num_classes=41
num_blocks=6
train_csv_path="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/train_protocol_mapped_all.csv"
trial_path="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/seen_test.txt"

# p1 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/seen_seen_same_speaker.txt"
# p2 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/seen_seen_diff_speaker.txt"
# p3 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/unseen_unseen_same_speaker.txt"
# p4 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/unseen_unseen_diff_speaker.txt"

input_layer=conv2d2
pos_enc_layer_type=rel_pos

save_dir=experiment/${encoder_name}_${num_blocks}_${embedding_dim}_${loss_name}

# ─────────────────────────── Setup ───────────────────────────
mkdir -p $save_dir
cp start.sh $save_dir
cp main.py $save_dir
cp -r module $save_dir
cp -r wenet $save_dir
cp -r loss $save_dir
echo "save_dir: $save_dir"

# ─────────────────────────── Launch ───────────────────────────
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
    --batch_size 200 \
    --num_workers 40 \
    --max_epochs 100 \
    --embedding_dim $embedding_dim \
    --save_dir $save_dir \
    --encoder_name $encoder_name \
    --train_csv_path $train_csv_path \
    --learning_rate 0.001 \
    --num_classes $num_classes \
    --trial_path $trial_path \
    --loss_name $loss_name \
    --num_blocks $num_blocks \
    --step_size 1 \
    --gamma 0.9 \
    --weight_decay 0.000001 \
    --input_layer $input_layer \
    --pos_enc_layer_type $pos_enc_layer_type
