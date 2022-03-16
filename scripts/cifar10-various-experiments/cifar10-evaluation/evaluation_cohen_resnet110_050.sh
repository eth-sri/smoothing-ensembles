CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py cifar10 \
    0.50 \
    output_dir/output_file \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7100.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7101.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7102.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7103.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7104.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7105.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7106.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7107.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7108.pth.tar \
    models-cifar10/cohen/resnet110/0.50/checkpoint-7109.pth.tar \
    --alpha 0.001 \
    --N 10000 \
    --skip 20 \
    --skip_offset 1 \
    --batch 5000
