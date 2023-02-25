import torch.nn as nn
import torch
from transformers import *
#from torchcrf import CRF

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
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
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

class BART_QG(nn.Module):
    def __init__(self,config, args):
        super().__init__()
        self.bart =  BartForConditionalGeneration.from_pretrained(config)


    def forward(self, input_ids,attention_mask=None,decoder_input_ids=None, labels=None):
        output = self.bart(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        return output

    def generate(self, batch, num_beams=5, do_sample=True):
        "encoder_outputs"
        encoder_ids= batch.encoder_ids
        encoder_attention_masks= batch.encoder_attention_masks
        decoder_ids=batch.decoder_ids

        model_kwargs = {
            "encoder_outputs": model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
                ),
            "decoder_input_ids":decoder_ids,
            "attention_mask":encoder_attention_masks,

        }

        encoded_ids = self.bart_QG.generate(num_beams=num_beams, do_sample=do_sample, **model_kwargs)
        return encoded_ids
