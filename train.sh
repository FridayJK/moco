data_dir="/workspace/mnt/storage/zhangjunkang/zjk2/data"
output_dir="./highway_output/18_7_pretrained_extendData_k256/"
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
    train_highway.py \
    --data-dir ${data_dir} \
    --dataset highway_s \
    --nce-k 256 \
    --batch-size 64 \
    --output-dir ${output_dir} \
    --base-learning-rate 0.01 \
    --crop 0.08 \
    --epochs 600