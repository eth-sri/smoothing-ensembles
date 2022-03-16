CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 1.00 \
    output_dir/output_file \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1209.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1203.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1206.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1204.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1202.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1207.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1205.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1201.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1200.pth.tar \
    models-cifar10/consistency/resnet110/1.00/checkpoint-1208.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
