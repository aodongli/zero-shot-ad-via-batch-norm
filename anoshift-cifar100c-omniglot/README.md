Please refer to `requirements.txt` for necessary libraries.

## Omniglot

1. Training

```
python Launch_Exps.py --dataset-name=omniglot --config-file=config_omniglot.yml
```

2. Testing 

```
for i in 0.01 0.05 0.1 0.2 # anomaly ratio
do
for j in 0 1 2 3 4 # repeat
do
python Launch_Exps.py --dataset-name=omniglot --config-file=config_omniglot.yml --qry-anomaly-ratio=$i --ckpt-path=RESULTS/omniglot/ckpt_10000.pt
done
done
```

## CIFAR100-C

1. Download [CIFAR-100-C](https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1) to the folder `data/cifar100/CIFAR-100-C/`

2. Training

```
python Launch_Exps.py --dataset-name=cifar100 --config-file=config_cifar100.yml
```

3. Testing

```
for k in gaussian_noise.npy shot_noise.npy impulse_noise.npy speckle_noise.npy motion_blur.npy zoom_blur.npy  gaussian_blur.npy glass_blur.npy  defocus_blur.npy  snow.npy  frost.npy fog.npy brightness.npy pixelate.npy saturate.npy spatter.npy contrast.npy elastic_transform.npy jpeg_compression.npy
do
echo $k
for i in 0.01 0.05 0.1 0.2
do
for j in 0 1 2 3 4
do
python Launch_Exps.py --dataset-name=cifar100 --corruption-file=$k --config-file=config_cifar100.yml --qry-anomaly-ratio=$i --ckpt-path=RESULTS/cifar100/ckpt_6000.pt
done
done
done
```

## AnoShift

1. Download [AnoShift](https://github.com/bit-ml/AnoShift) to folder `data/anoshift/`

2. Training

```
python Launch_Exps.py
```

3. Testing

```
for i in 0.01 0.05 0.1 0.2 # anomaly ratio
do
for j in 0 1 2 3 4 # repeat
do
python Launch_Exps.py --qry-anomaly-ratio=$i --ckpt-path=RESULTS/anoshift/ckpt_1000.pt
done
done
```
