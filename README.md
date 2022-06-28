Dual Stream Transformer for Generic Event Boundary Captioning
=====
### Prerequisites
1. Clone this repository

```
git clone https://github.com/GX77/Dual-Stream-Transformer-for-Generic-Event-Boundary-Captioning.git
```

2. Download Kinetic-GEBC Dataset

3. Install dependencies

​		Python 3.7、PyTorch 1.1.0


### Training and Inference
1. Training

```
python3 train.py \
--dset_name kin \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--region_feature_dir ${r_feat_dir} \
--region_number ${r_n} \
--word2idx_path ${word2idx_path} \
```
2. Inference 

```
python3 translate.py
```
## Others
This code used resources from the following projects: 
[Transformers](https://github.com/huggingface/transformers), [Mart](https://github.com/GX77/recurrent-transformer)

