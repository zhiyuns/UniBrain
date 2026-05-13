GPU_NUM=$1
N_NODES=$2
NODE_RANDK=$3

torchrun \
  --nnodes=$N_NODES \
  --node_rank=$NODE_RANDK \
  --nproc_per_node=$GPU_NUM \
  --master_addr=127.0.0.1 \
  --master_port=7730 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./data/configs/RadGenome.yaml \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume False \
  --resume_from results/checkpoints_s1/0002000 \
  --resume-model-only True \
  --finetune-from-ema False \
  --use_ema True \
  --log_every 1 \
  --peak_device_tflops 0 \
  --lr 2e-5 \
  --num_worker 1 \
  --gradient_accumulation_steps 1 \
  --num_shard $GPU_NUM \
  --total_steps 2000 \
  --warmup_steps 200 \
  --save_every 1000 \
  --visualization_interval 100 \
  --visual_gen True \
  --visual_und True \
  --freeze_llm False \
  --freeze_vit False \
  --freeze_und False \
  --freeze_gen False \
  --wandb_offline True \
  --sharding_strategy HYBRID_SHARD \
  --use_flex True \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.3 \
  --vit_cond_dropout_prob 0.3 \
  --ema 0.995 \