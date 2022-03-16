CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 0.25 \
    output_dir/output_file \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1002.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1006.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1009.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1005.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1003.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1004.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1001.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1008.pth.tar \
    models-cifar10/consistency/resnet110/0.25/checkpoint-1007.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
