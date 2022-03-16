CUDA_VISIBLE_DEVICES=0 python certify_ensemble_denoising.py \
    cifar10 \
    0.25 \
    output_dir/output_file \
    models-cifar10/denoised-smoothing/checkpoint-ResNet110_90epochs-noise_0.00.pth.tar \
    models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_3-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_4-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_5-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_1-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
