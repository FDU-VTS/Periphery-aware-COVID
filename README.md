# Periphery-aware COVID-19 Diagnosis with Contrastive Representation Enhancement

This repo covers a PyTorch implementation for the following paper, using the [CC-CCII dataset](http://ncov-ai.big.ac.cn/download?lang=en) as an illustrative example:  
Periphery-aware COVID-19 Diagnosis with Contrastive Representation Enhancement. 

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs.

**(1) PSP Pre-training**
```
cd ./PSP
python train.py --batch_size 64 --epoch 50 --lr 1e-4
```
**(2) Pneumonia Classification with CRE**  
```
CRE is coming soon...
```

## Reference
```

```
