# Post Training Stage
## 1. finetune with custom-weighted loss function
1. cd internvl_chat
2. Start post training: `CUDA_VISIBLE_DEVICES=xxx bash shell/post_tune.sh`
3. Merge lora: `python tools/merge_lora.py xxx`

## 2. Use DPO to further boost performance
`NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=xxx swift rlhf --rlhf_type dpo --model_type internvl2_5 --num_train_epochs 1 --dataset xxx --model xxx  --deepspeed zero2 --eval_steps 20000 --save_steps 20000 --train_type lora --learning_rate 1e-4`