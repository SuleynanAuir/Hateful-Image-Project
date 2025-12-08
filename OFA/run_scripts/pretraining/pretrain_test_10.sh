#!/bin/bash

# OFA Pretraining - 10 samples test

export MASTER_PORT=1052
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

bpe_dir=../../utils/BPE
user_dir=../../ofa_module
restore_file=../../checkpoints/caption_base_best.pt

data_dir=../../dataset/pretrain_data
neg_sample_dir=${data_dir}/negative_sample
data=${data_dir}/vision_language_examples.tsv

selected_cols=0,1,2,3,4,5,6,7

task=unify_task
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=1e-5
max_epoch=3
warmup_ratio=0.1
batch_size=1
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=30
num_bins=1000
patch_image_size=256
sample_patch_num=196
max_image_size=512

save_path=./checkpoints_test

python ../../train.py \
  $data \
  --selected-cols=${selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --neg-sample-dir=${neg_sample_dir} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=1 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=3 \
  --save-interval=1 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --num-workers=0
