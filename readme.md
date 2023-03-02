## Question Generation Task for FairyTale Dataset
## conda install
```
conda create --name fairytale python=3.10.8
source activate fairytale
conda install pytorch=1.13.1 torchvision==0.14.1  pytorch-cuda=11.6 -c pytorch -c nvidia
pip install scikit-learn
pip install tensorboard_logger
pip install pandas
```

### run the training over 2 gpus
```bash
python -m torch.distributed.launch --nproc_per_node=2 run_QG.py \
    --model_name_or_path /media/abhidip/2F1499756FA9B1151/QG/model/bart-base/ \
    --do_train \
    --csv_file /media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/train_split.csv \
    --num_train_epochs 50 \
    --save_steps 750 \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --task ask_question
```

### Inference
```bash
python run_QG.py \
    --do_test \
    --model_name_or_path /media/abhidip/2F1499756FA9B1151/QG/model/checkpoint-0-1500 \
    --csv_file /media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/test_split.csv \
    --task ask_question
```

### evaluation
