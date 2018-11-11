import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batchFlag=True, dropProb=0.4):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(dropProb)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=batchFlag, dropout=dropProb)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        pass
    
    def forward(self, features, captions):
        # output, (h_n, c_n)
        captions = captions[:, :-1]
        embed = self.embed(captions)
        allParams = torch.cat([features.unsqueeze(1), embed], 1)
        output ,(h_n, c_n) = self.lstm(allParams)
        op = self.fc(output)
        return op

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            op, states = self.lstm(inputs, states)   # New states
            out = self.fc(op.squeeze(1))
            argmax = out.max(1)
            index = argmax[1].item()
            tokens.append(index)
            inputs = self.embed(argmax[1].long()).unsqueeze(1)   # new inputs
            if index == 1:  # <end>
                break
        return tokens