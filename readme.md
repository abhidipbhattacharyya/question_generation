## conda install
```
conda create --name fairytale python=3.10.8
source activate fairytale
conda install pytorch=1.13.1 torchvision==0.14.1  pytorch-cuda=11.6 -c pytorch -c nvidia
pip install scikit-learn
pip install tensorboard_logger
pip install pandas
```
## Question Generation Task for FairyTale Dataset
### run the training over 2 gpus

on ego server
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_QG.py \
    --model_name_or_path /data/abhidip/QG_data/experiments/model/bart-base/ \
    --do_train \
    --csv_file /data/abhidip/QG_data/CU-stroy-QA-Data/train_split.csv \
    --num_train_epochs 50 \
    --save_steps 750 \
    --batch_size 4 \
    --learning_rate 3e-4 \
    --task ask_question
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_QG.py --do_train --model_name_or_path /data/abhidip/QG_data/experiments/model/bart-base/ --csv_file /data/abhidip/QG_data/CU-stroy-QA-Data/train_split.csv --num_train_epochs 50 --save_steps 750 --batch_size 4 --learning_rate 3e-4 --task ask_question
```
### Inference

```bash
CUDA_VISIBLE_DEVICES=2 python run_QG.py --do_test --csv_file /data/abhidip/QG_data/CU-stroy-QA-Data/test_split.csv --task ask_question --model_name_or_path /data/abhidip/QG_data/experiments/model/
```


### evaluation
