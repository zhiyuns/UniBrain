bash scripts/train_s1.sh 8 1 0
rm -rf results/checkpoints/0002000/ema.safetensors # never use ema checkpoint in this stage!!!
mv results/checkpoints results/checkpoints_s1
bash scripts/train_s2.sh 8 1 0
mv results/checkpoints results/checkpoints_s2
bash scripts/train_s3.sh 8 1 0
mv results/checkpoints results/checkpoints_s3