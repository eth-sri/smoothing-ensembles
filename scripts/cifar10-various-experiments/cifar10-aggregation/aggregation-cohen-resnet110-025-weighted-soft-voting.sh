CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 \
    0.25 \
    output_dir/output_file \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7000.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7001.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7002.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7003.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7004.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7005.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7006.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7007.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7008.pth.tar \
    models-cifar10/cohen/resnet110/0.25/checkpoint-7009.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000 \
    --aggregation_scheme 3
