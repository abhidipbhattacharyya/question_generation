import torch.nn as nn
import torch
from transformers import BartForConditionalGeneration


#from torchcrf import CRF



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
