CUDA_VISIBLE_DEVICES=0 python certify_ensemble.py \
    cifar10 \
    1.00 \
    output_dir/output_file \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2233.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2203.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2243.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2211.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2206.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2231.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2207.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2241.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2202.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2249.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2214.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2247.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2238.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2221.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2229.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2242.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2237.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2232.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2210.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2219.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2200.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2228.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2227.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2213.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2205.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2245.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2239.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2230.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2220.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2217.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2234.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2226.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2208.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2235.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2225.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2216.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2209.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2246.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2244.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2212.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2201.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2222.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2236.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2248.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2224.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2223.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2218.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2240.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2204.pth.tar \
    models-cifar10/consistency/resnet20/1.00/checkpoint-2215.pth.tar \
    --alpha 0.001 \
    --N 100000 \
    --skip 20 \
    --batch 1000
