import torch.nn as nn
import torch
from transformers import BartForConditionalGeneration


#from torchcrf import CRF



class BART_QG(nn.Module):
    def __init__(self,config, args):
        super().__init__()
        self.bart =  BartForConditionalGeneration.from_pretrained(config)


    def forward(self, input_ids,attention_mask=None,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        output = self.bart(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=labels)
        return output

    def generate(self, input_ids,attention_mask=None,decoder_input_ids=None, decoder_attention_mask=None, num_beams=5, do_sample=True):
        "encoder_outputs"
        encoder_ids= input_ids
        encoder_attention_masks= attention_mask
        decoder_ids=decoder_input_ids
        #print(encoder_ids.size())
        #print(decoder_ids.size())
        #print("hgere")
        #print(self.bart.config.is_encoder_decoder)
        model_kwargs = {
            "encoder_outputs": self.bart.get_encoder()(
                encoder_ids, attention_mask=encoder_attention_masks, return_dict=True
                ),
            "decoder_input_ids":decoder_ids,
            "attention_mask":encoder_attention_masks,
            "max_length":128,
            "decoder_attention_mask":decoder_attention_mask,
        }

        encoded_ids = self.bart.generate(num_beams = num_beams,  do_sample=do_sample, **model_kwargs)

        print("resturn seq {}".format(encoded_ids.size()))

        return encoded_ids
