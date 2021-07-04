# Periphery-aware COVID-19 Diagnosis with Contrastive Representation Enhancement

This repo covers a PyTorch implementation for the paper "Periphery-aware COVID-19 Diagnosis with Contrastive Representation Enhancement", using the [CC-CCII dataset](http://ncov-ai.big.ac.cn/download?lang=en) as an illustrative example.  

## Installation
* Install Pytorch 1.7.1 and CUDA 10.1
* Clone this repo
```
git clone https://github.com/FDU-VTS/Periphery-aware-COVID
cd Periphery-aware-COVID
```

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs.

**(1) PSP Pre-training**
```
cd PSP
python train.py --batch_size 64 --epoch 50 --lr 1e-4
```
**(2) Pneumonia Classification with CRE**  
```
cd CRE
python main_supcon.py --batch_size 4 --epoch 50 --lr 1e-4
```

## Reference
```
@article{2021Periphery,
  title={Periphery-aware COVID-19 Diagnosis with Contrastive Representation Enhancement},
  author={ Hou, J.  and  Xu, J.  and  Jiang, L.  and  Du, S.  and  Xue, X. },
  journal={Pattern Recognition},
  volume={118},
  number={2},
  pages={108005},
  year={2021},
}
```
