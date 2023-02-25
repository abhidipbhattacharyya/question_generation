from transformers import BartTokenizer
import argparse
import base64
import numpy as np
import os
import os.path as op
import random, time, json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, BartConfig
import tensorboard_logger as tb_logger
import logging
from utils import LogCollector
from qg_dataset import build_dataset

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'#, world_size=world_size, rank = local_rank
        )
    return local_rank

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, csv_file, tokenizer, is_distributed=True,is_train=True):
    dataset = build_dataset(args,  tokenizer, is_train=is_train)
    if is_train:
        shuffle = True
        images_per_gpu = args.batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_epochs
        #logger.info("Train with {} images per GPU.".format(images_per_gpu))
        #logger.info("Total batch size {}".format(images_per_batch))
        #logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader

def train(args, train_dataloader, model, tokenizer):
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logging.info("***** Running training *****")
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            encoder_ids= batch.encoder_ids.to(args.device)
            encoder_attention_masks=batch.encoder_attention_masks.to(args.device)
            decoder_ids=batch.decoder_ids.to(args.device)
            target_ids=batch.target_ids.to(args.device)
            model.train()

            outputs = model(input_ids=encoder_ids,
                attention_mask=encoder_attention_masks,
                decoder_input_ids=decoder_ids,
                labels=target_ids
            )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logging.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch, global_step,
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step,
                        batch_acc, global_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                    logging.info("model saved at {}".format(checkpoint_dir))

    return checkpoint_dir





def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', default='/media/abhidip/2F1499756FA9B1151/data/flickr/entity/flickr30k_entities-master/',
                        help='path to datasets')

    parser.add_argument('--model_name_or_path', default='/media/abhidip/2F1499756FA9B1151/SCAN_models/GOT/bert-base-uncase',
                        help='Path to bert base.')

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")

    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_train_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')

    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")

    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=1000, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=10, type=int,
                        help='Number of steps to run validation.')

    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument('--output_dir', default='/media/abhidip/2F1499756FA9B1151/SCAN_models/GOT/',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='/media/abhidip/2F1499756FA9B1151/SCAN_models/GOT/log', help='Path to save Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluation.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")

    parser.add_argument("--max_encoder_input_length", default=1024, type=int,
                        help="The maximum sequence length for encoder.")
    parser.add_argument("--max_decoder_input_length", default=128, type=int,
                        help="The maximum sequence length for decoder.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum sequence length for generation.")

    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")


    args = parser.parse_args()
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    set_seed(args.seed, args.num_gpus)

    checkpoint = args.model_name_or_path
    #config = BartConfig.from_pretrained(checkpoint)
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    model = load_checkpoint(checkpoint, args)#checkpoint, config=config)
    model.to(args.device)

    if args.do_train:
        train_dataloader = make_data_loader(args, args.csv_file, tokenizer,
            args.distributed, is_train=True)
        last_checkpoint = train(args, train_dataloader, model, tokenizer)
        print("training done. Last chkpt {}".format(last_checkpoint))
