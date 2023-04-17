## Question Generation Task for FairyTale Dataset
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





### run training for TQA data

```bash
python -m torch.distributed.launch --nproc_per_node=2 run_tqa.py \
    --model_name_or_path /media/abhidip/2F1499756FA9B1151/QG/model/bart-base/ \
    --do_train \
    --csv_file /media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/tqa/tqa_train.csv \
    --num_train_epochs 50 \
    --save_steps 4182 \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --task ask_question
```
