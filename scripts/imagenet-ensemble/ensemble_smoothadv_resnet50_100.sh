IMAGENET_DIR=PATH \
    CUDA_VISIBLE_DEVICES=0,1 \
    python certify_ensemble.py \
    imagenet \
    1.00 \
    output_dir/output_file \
    models-imagenet/smoothadv-resnet50-100/checkpoint-PGD1step-eps512-100.pth.tar \
    models-imagenet/smoothadv-resnet50-100/checkpoint-PGD1step-eps1024-100.pth.tar \
    models-imagenet/smoothadv-resnet50-100/checkpoint-PGD1step-eps256-100.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 100 \
    --batch 750 \
    --center_layer 1
