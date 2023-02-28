## run the training over 2 gpus
`python -m torch.distributed.launch --nproc_per_node=2 run_QG.py --do_train`

python run_QG.py --do_test --model_name_or_path /media/abhidip/2F1499756FA9B1151/QG/model/checkpoint-0-1500 --csv_file /media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/test_split.csv
