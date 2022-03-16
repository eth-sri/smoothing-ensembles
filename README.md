# Boosting Randomized Smoothing with Variance Reduced Classifiers <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This it the official codebase for the paper [Boosting Randomized Smoothing with Variance Reduced Classifiers (ICLR 2022)](https://www.sri.inf.ethz.ch/publications/horvat2021boosting).

In the paper, we show -- theoretically and empirically -- that **ensembles** reduce variance under Randomized Smoothing, yielding higher certified accuracy, leading to a new state-of-the-art on `CIFAR-10` and `ImageNet`. To reduce computational overhead, we additionally introduce a more efficient aggregation mechanism for ensembles (**K-Consensus Aggregation**), and an adaptive sampling scheme reducing the average sample requirement for certification for Randomized Smoothing in the predetermined radius setting (**Adaptive Sampling**).

## Getting Started

To run things, first please install the required environment:

```
conda create --name smoothing-ensembles python=3.8
conda activate smoothing-ensembles
pip install -r requirements.txt
```

To install pytorch 1.8.0 and torchvision 0.9.0 you can use the following command (depending on your installed CUDA version which can be checked e.g., by running `nvidia-smi`):
```
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
If you are not using the most recent GPU drivers, please see here (https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for a compatible cudatoolkit version.

Models can be trained using the code published along the corresponding training methods and by following their instructions. 
Namely, https://github.com/locuslab/smoothing for Gaussian, https://github.com/jh-jeong/smoothing-consistency for Consistency and https://github.com/Hadisalman/smoothing-adversarial for SmoothAdv. 
The exact parameters and commands we use are detailed in `training.md`.

Alternatively, we provide the trained models we use here: https://mega.nz/file/zeQlVQAS#p48AffIzrl8lOHki5r6P4UwAy3GZt-1TmxKO0q5UHOI. 
Please make sure to move the folders `models-cifar10` and `models-imagenet` into the root directory of this codebase to run the scripts.
This can be done by executing the following commands after downloading the models to your `~/Downloads` directory:
```
tar -xf ~/Downloads/models_smoothing_ensembles.tar.gz -C ./
mv models_smoothing_ensembles/* ./.
rm models_smoothing_ensembles/ -r
```

## Examples

Here are some basic examples of how to use this codebase.
More scripts to reproduce our experiments are in the folder `scripts`.

### Ensembles

We can certify robustness for ensembles via `certify_ensemble.py`.
Here is a basic example for a base classifier which is an ensemble consisting of 3 single models:

```
python certify_ensemble.py \
    cifar10 \
    0.25 \
    ./output_dir/output_file \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1001.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1002.pth.tar \
    --alpha 0.001 \
    --N0 100 \
    --N 100000 \
    --skip 20 \
    --batch 1000
```

The first argument is the dataset (here `cifar10`), the second is the noise with which we do the certification (here `sigma = 0.25`) and the third is the path to the output files (here `output_dir/output_file`). Note that the `output_file_i` will save the output file for the `i/2`-th single model for even `i`, and the output file of the ensemble of the first `(i+1)/2` single models for odd `i`. The following arguments are the paths to the models, we can have as many models as we want after each other. The remaning arguments are optional.

### Ensembles via K-Consensus

In addition to ensembles, here we also apply K-Consensus Aggregation, i.e. if the first K models agree on the class of a perturbed sample, we stop.
The command is the same as for standard ensembles with with the exception of the additional argument `--voting_size` which determines K and the fact that `output_file` only saves the output of the whole ensemble, i.e. we don't save results of each individual model and smaller ensembles.

```
python certify_ensemble_k_consensus.py \
    cifar10 \
    0.25 \
    ./output_dir/output_file \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1001.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1002.pth.tar \
    --alpha 0.001 \
    --N0 100 \
    --N 100000 \
    --skip 20 \
    --batch 1000 \
    --voting_size 2
```

### Adaptive Sampling

Now we provide an example of Randomized Smoothing via Adaptive Sampling using `certify_ensemble_adaptive.py`.
The key interface change compared to `certify_ensemble.py` is that now we have the additional argument `radius_to_certify` which is the predetermined radius we want to consider for certification and the numbers `Ni` (for currently `i` from `{1, 2, 3, 4, 5}`) determine how many samples should be used in the `i`-th step.

```
python certify_ensemble_adaptive.py \
    cifar10 \
    0.25 \
    ./output_dir/output_file \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar \
    --alpha 0.001 \
    --N0 100 \
    --skip 20 \
    --batch 1000 \
    --beta 0.0001 \
    --radius_to_certify 0.25 \
    --N1 1000 \
    --N2 10000 \
    --N3 125000 
```

Note that similarly to above, we can list an arbitrary number of models if we want to consider an ensemble as a base classifier.

### Ensembles via K-Consensus and Adaptive Sampling

The file `certify_ensemble_adaptive_and_k_consensus.py` combines all the previous ideas and supports Randomized Smoothing via ensembles, and certification time reduction via K-Consensus Aggregation and Adaptive Sampling. 
The arguments are essentially the union of the arguments from `certify_adaptive_k_consensus.py` and `certify_ensemble_adaptive.py`.

```
python certify_ensemble_adaptive_and_k_consensus.py \
    cifar10 \
    0.25 \
    ./output_dir/output_file \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1001.pth.tar \
    ./models-cifar10/consistency/resnet110/0.25/checkpoint-1002.pth.tar \
    --alpha 0.001 \
    --N0 100 \
    --N 100000 \
    --skip 20 \
    --batch 1000 \
    --beta 0.0001 \
    --voting_size 2 \
    --radius_to_certify 0.25 \
    --N1 1000 \
    --N2 10000 \
    --N3 125000
```

### Ensembles for Denoised Smoothing

Here, we give an example of ensembles for Denoised Smoothing via `certify_ensemble_denoising.py`. The key difference to `certify_ensemble.py` is that here, we first have to provide the classifier trained on unperturbed samples. Then we list the denoisers.

```
python certify_ensemble_denoising.py \
    cifar10 \
    0.25 \
    ./output_dir/output_file \
    ./models-cifar10/denoised-smoothing/checkpoint-ResNet110_90epochs-noise_0.00.pth.tar \
    ./models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_3-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    ./models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_4-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    ./models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_5-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    ./models-cifar10/denoised-smoothing/checkpoint-stab_obj-cifar10_smoothness_obj_adamThenSgd_1-resnet110_90epochs-dncnn_wide-noise_0.25.pth.tar \
    --alpha 0.001 \
    --N0 100 \
    --N 100000 \
    --skip 20 \
    --batch 1000
```

## Contributors

- Miklós Z. Horváth.
- [Mark Niklas Mueller](https://www.sri.inf.ethz.ch/people/mark)
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## Citation

If you find this work useful for your research, please cite it as:

```
@inproceedings{
    horvath2022boosting,
    title={Boosting Randomized Smoothing with Variance Reduced Classifiers},
    author={Mikl{\'o}s Z. Horv{\'a}th and Mark Niklas Mueller and Marc Fischer and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=mHu2vIds_-b}
}
```
