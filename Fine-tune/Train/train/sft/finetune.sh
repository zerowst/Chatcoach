output_model=../../model_llama/output_path_match_inst
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
#cp ./finetune_other.sh ${output_model}
deepspeed --num_gpus=4 --master_port 20001 finetune_clm_lora.py \
    --model_name_or_path FlagAlpha/Llama2-Chinese-7b-Chat \
    --train_files ../../data/match_inst_train.csv \
    --validation_files  ../../data/match_inst_eval.csv \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 400 \
    --learning_rate 2e-3 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 2 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20 \
    --eval_steps 10 \
    --save_total_limit 20 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to wandb \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
