import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch as torch
import torch.nn.functional as F


# Sigmoid + Cross + Tied
class ProjLanguageContextCharLinguistic(nn.Module):
    def __init__(self, input_size, layer1_size, kernel_num, char_vec_size, filter_withs):
        super(ProjLanguageContextCharLinguistic, self).__init__()

        self._cnn_char = nn.ModuleList([nn.Conv2d(1, kernel_num, (w, char_vec_size)) for w in filter_withs])
        character_size = kernel_num * len(filter_withs)
        self.char_bn = nn.BatchNorm1d(character_size, momentum=0.01)

        self.encoder = nn.Parameter(torch.Tensor(input_size, layer1_size).uniform_(0.0, 0.02))
        self.encoder_context = nn.Parameter(torch.Tensor(input_size, layer1_size).uniform_(0.0, 0.02))

        self.encode_bias = nn.Parameter(torch.Tensor(layer1_size).uniform_(0.0, 0.02))
        self.decode_bias = nn.Parameter(torch.Tensor(input_size).uniform_(0.0, 0.02))
        self.decode_context_bias = nn.Parameter(torch.Tensor(input_size).uniform_(0.0, 0.02))
        # self.decode_char_bias = nn.Parameter(torch.Tensor(character_size).uniform_(0.0, 0.02))

        self.encoder_bn = nn.BatchNorm1d(layer1_size, momentum=0.01)
        self.decoder_bn = nn.BatchNorm1d(input_size, momentum=0.01)
        self.decoder_context_bn = nn.BatchNorm1d(input_size, momentum=0.01)
        # self.decoder_char_bn = nn.BatchNorm1d(character_size, momentum=0.01)

    def forward(self, x, x_context=None, input_chars=None, cross_encoded=None):
        if x_context is None and input_chars is None and cross_encoded is None:
            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + self.encode_bias)
            encoded = self.encoder_bn(encoded)
            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)
            return encoded, decoded
        elif x_context is None and input_chars is None and cross_encoded is not None:
            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + self.encode_bias)
            encoded = self.encoder_bn(encoded)
            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)

            cross_decoded = torch.sigmoid(torch.mm(cross_encoded, torch.transpose(self.encoder, 0, 1)) +
                                          self.decode_bias)
            cross_decoded = self.decoder_bn(cross_decoded)
            return encoded, decoded, cross_decoded
        else:
            input_chars = input_chars.view(input_chars.size(0), 1, input_chars.size(1), input_chars.size(2))
            conv_chars = [F.relu(conv(input_chars)).squeeze(3) for conv in self._cnn_char]
            conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]  # [(N,Co), ...]*len(Ks)
            conv_chars = torch.cat(conv_chars, 1)  # dim1: batchsize  dim2: 100

            conv_chars = self.char_bn(conv_chars)

            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + torch.mm(x_context, self.encoder_context) + self.encode_bias)
            encoded = self.encoder_bn(encoded)

            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)
            decoded_context = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder_context, 0, 1)) +
                                            self.decode_context_bias)
            decoded_context = self.decoder_context_bn(decoded_context)

            if cross_encoded is not None:
                cross_decoded = torch.sigmoid(torch.mm(cross_encoded, torch.transpose(self.encoder, 0, 1)) +
                                              self.decode_bias)
                cross_decoded = self.decoder_bn(cross_decoded)
                cross_decoded_context = torch.sigmoid(torch.mm(cross_encoded, torch.transpose(self.encoder_context, 0, 1))
                                                      + self.decode_context_bias)
                cross_decoded_context = self.decoder_context_bn(cross_decoded_context)

                return encoded, conv_chars, decoded, decoded_context, cross_decoded, cross_decoded_context

            return encoded, conv_chars, decoded, decoded_context
