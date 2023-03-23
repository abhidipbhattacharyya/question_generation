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
from qg_dataset_1 import build_dataset
#from qg_dataset import build_dataset
from model import  BART_QG
from utils import mkdir, csv_writer, concat_csv_files, delete_csv_files, reorder_csv_keys

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
    dataset = build_dataset(args,  tokenizer, is_train=True)
    if is_train:
        shuffle = True
        images_per_gpu = args.batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
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
        collate_fn=dataset.collate_fn,
    )
    return data_loader

def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    bart = model_to_save.bart
    for i in range(num_trial):
        try:
            bart.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logging.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def load_checkpoint(checkpoint, args, logger=None):
    #checkpoint = args.eval_model_dir
    assert op.isdir(checkpoint)
        #config = config_class.from_pretrained(checkpoint)
        #config.output_hidden_states = args.output_hidden_states
        #self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        #logger.info("Evaluate the following checkpoint: %s", checkpoint)
    bart_QG = BART_QG(checkpoint, args)
    return bart_QG

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
        scheduler = get_constant_schedule_with_warmup(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps =t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logging.info("***** Running training *****")
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Batch size per GPU = %d", args.batch_size)
    logging.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.batch_size * get_world_size() * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for bstep, batch in enumerate(train_dataloader):
            #print(batch)
            encoder_ids= batch.encoder_ids.to(args.device)
            encoder_attention_masks=batch.encoder_attention_masks.to(args.device)
            decoder_ids=batch.decoder_ids.to(args.device)
            target_ids=batch.target_ids.to(args.device)
            decoder_attention_masks= batch.decoder_attention_masks.to(args.device)
            model.train()

            outputs = model(input_ids=encoder_ids,
                attention_mask=encoder_attention_masks,
                decoder_input_ids=decoder_ids,
                decoder_attention_mask= decoder_attention_masks,
                labels=target_ids
            )
            loss, logits = outputs[:2]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            if (bstep + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logging.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step,
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step)
                    )
                    if is_main_process():
                        tb_logger.log_value('epoch',epoch,step=global_step)
                        tb_logger.log_value('Eiter',global_step,step=global_step)
                        tb_logger.log_value('lr',optimizer.param_groups[0]["lr"],step=global_step)
                        tb_logger.log_value('loss_Avg',global_loss / global_step,step=global_step)
                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                    logging.info("model saved at {}".format(checkpoint_dir))

    return checkpoint_dir

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def generate_output_sequences_for_comp_data(model, data_loader, predict_file, tokenizer,args = None, with_attribute=False):
    if not (args.do_test or args.do_eval):
        return output_file

    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(),
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    def gen_rows():
        with torch.no_grad():
            for bstep, batch in enumerate(data_loader):
                #print(batch)
                pair_ids = batch.pair_ids
                encoder_ids= batch.encoder_ids.to(args.device)
                encoder_attention_masks=batch.encoder_attention_masks.to(args.device)
                decoder_ids=batch.decoder_ids.to(args.device)
                decoder_attention_masks= batch.decoder_attention_masks.to(args.device)
                target_ids=batch.target_ids.to(args.device)
                encoded_op = model.generate(encoder_ids,attention_mask=encoder_attention_masks,decoder_input_ids=decoder_ids,decoder_attention_mask=decoder_attention_masks)
                decoded_outputs = tokenizer.batch_decode(encoded_op, skip_special_tokens=True)
                decoder_ip = tokenizer.batch_decode(decoder_ids, skip_special_tokens=True)
                for pair_id, dec_op, di in zip(pair_ids, decoded_outputs, decoder_ip):
                    dec_op1 = dec_op.replace(di,"")
                    if isinstance(pair_id, torch.Tensor):
                        pair_id = pair_id.item()
                    yield pair_id,di,dec_op1
    csv_writer(gen_rows(), cache_file)
    if world_size > 1:
        torch.distributed.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_csv_files(cache_files, predict_file)
        delete_csv_files(cache_files)
        reorder_csv_keys(predict_file, data_loader.dataset.pair_ids, predict_file)
        print("op is at {}".format(predict_file))
    if world_size > 1:
        torch.distributed.barrier()

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', default='/media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/train_split.csv',
                        help='path to datasets')

    parser.add_argument('--model_name_or_path', default='/media/abhidip/2F1499756FA9B1151/QG/model/bart-base/',
                        help='Path to bert base.')

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")

    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_train_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=4, type=int,
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
    parser.add_argument('--output_dir', default='/media/abhidip/2F1499756FA9B1151/QG/model',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='/media/abhidip/2F1499756FA9B1151/QG/log', help='Path to save Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluation.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")

    parser.add_argument("--max_encoder_input_length", default=1024, type=int,
                        help="The maximum sequence length for encoder.")
    parser.add_argument("--max_decoder_input_length", default=64, type=int,
                        help="The maximum sequence length for decoder.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum sequence length for generation.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--task', default="both",help='ask_question|ask_answer|both')
    parser.add_argument('--predict_file', default='/media/abhidip/2F1499756FA9B1151/QG/model/predict_m4_bothTrModel.csv',
                        help='Path to save the generated op.')

    args = parser.parse_args()
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    set_seed(args.seed, args.num_gpus)


    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(args.logger_name, flush_secs=5)

    checkpoint = args.model_name_or_path
    #config = BartConfig.from_pretrained(checkpoint)
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    logging.info("Evaluate the following checkpoint: %s", checkpoint)
    model = load_checkpoint(checkpoint, args)#checkpoint, config=config)
    print("b4====config v_size:{} model_vsize:{}".format(model.bart.config.vocab_size, model.bart.model.shared.num_embeddings))
    if model.bart.config.vocab_size == 50265 and model.bart.model.shared.num_embeddings==50265:
        attribute_tok = ["action:", "prediction:", "feeling:", "outcome resolution:", "character:", "causal relationship:", "setting:"]
        additional_special_tokens = ["ask_question:", "ask_answer:", "context:","<a>","</a>","<q>","</q>", "both:" ]+attribute_tok
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        desired_vsize = model.bart.model.shared.num_embeddings + num_added_toks
        model.bart.config.vocab_size = desired_vsize
        model.bart.resize_token_embeddings(desired_vsize)

    print("config v_size:{} model_vsize:{}".format(model.bart.config.vocab_size, model.bart.model.shared.num_embeddings))
    model.to(args.device)

    if args.do_train:
        train_dataloader = make_data_loader(args, args.csv_file, tokenizer,
            args.distributed, is_train=True)
        last_checkpoint = train(args, train_dataloader, model, tokenizer)
        print("training done. Last chkpt {}".format(last_checkpoint))

    elif args.do_test:
        print("calling this")
        test_dataloader = make_data_loader(args, args.csv_file, tokenizer,
            args.distributed, is_train=False)
        generate_output_sequences_for_comp_data(model, test_dataloader, args.predict_file, tokenizer, args = args)

if __name__ == "__main__":
    main()
