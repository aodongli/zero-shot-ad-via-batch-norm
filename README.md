# Zero-Shot Anomaly Detection via Batch Normalization

Official code repository for NeurIPS 2023 paper [Zero-Shot Anomaly Detection via Batch Normalization](https://arxiv.org/abs/2302.07849).

Code for different datasets is shown in the folder names. Refer to each folder for the datasets of interest. 
- [MVTec AD](https://github.com/aodongli/zero-shot-ad-via-batch-norm/tree/main/mvtec-ad)
- [AnoShift, CIFAR100-C, Omniglot](https://github.com/aodongli/zero-shot-ad-via-batch-norm/tree/main/anoshift-cifar100c-omniglot)

Package requirements are listed in each folder's `requirements.txt`. Run `pip install -r requireme.txt` to install all packages.

## Brief Introduction
We introduce a straightforward yet powerful approach to zero-shot anomaly detection. This method requires minimal configurations: 1) ensure that the deep model is set for batch-level prediction and 2) maintain all batch normalization layers in the training mode during inference. Below, you'll find a step-by-step comparison with the traditional *stationary* anomaly detection framework. Key configurations are color-highlighted for clarity. 

<img title="" src="./acr-diff.png" alt="acr" data-align="inline">

---------
```
@inproceedings{acr,
  title={Zero-Shot Anomaly Detection via Batch Normalization},
  author={Li, Aodong and Qiu, Chen and Kloft, Marius and Smyth, Padhraic and Rudolph, Maja and Mandt, Stephan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
