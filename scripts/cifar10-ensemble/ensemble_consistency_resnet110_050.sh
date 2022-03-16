CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 0.50 \
    output_dir/output_file \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1106.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1104.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1109.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1101.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1102.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1100.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1108.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1107.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1105.pth.tar \
    models-cifar10/consistency/resnet110/0.50/checkpoint-1103.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
