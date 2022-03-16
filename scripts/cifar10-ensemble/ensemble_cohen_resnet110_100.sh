CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 1.00 \
    output_dir/output_file \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7207.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7206.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7203.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7200.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7205.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7201.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7208.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7209.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7202.pth.tar \
    models-cifar10/cohen/resnet110/1.00/checkpoint-7204.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
