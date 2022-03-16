IMAGENET_DIR=PATH \
    CUDA_VISIBLE_DEVICES=0,1 \
    python certify_ensemble.py \
    imagenet \
    1.00 \
    output_dir/output_file \
    models-imagenet/consistency-resnet50-100/checkpoint-0.pth.tar \
    models-imagenet/consistency-resnet50-100/checkpoint-1.pth.tar \
    models-imagenet/consistency-resnet50-100/checkpoint-2.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 100 \
    --batch 750
