import torch
import os


def load_model(weight_path, num):
    """
    returns state_dict of encoder, decoder, att_decoder
    return: (encoder, decoder, att_decoder)
    """
    encoder = torch.load(os.path.join(weight_path, "encoder-{}.pth".format(num)))
    decoder = torch.load(os.path.join(weight_path, "decoder-{}.pth".format(num)))
    att_decoder = torch.load(os.path.join(weight_path, "attention_decoder-{}".format(num)))
    return encoder, decoder, att_decoder
