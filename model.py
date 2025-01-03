import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, XLNetModel
import math
import copy

import sys

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')      ## nn.NLLLoss
        '''
        nn.NLLLoss
        官方文档中介绍称： nn.NLLLoss输入是一个对数概率向量和一个目标标签，它与nn.CrossEntropyLoss的关系可以描述为：softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
        '''

    def forward(self, pred, target, mask):
        '''
        param pred: (batch_size, num_utterances, n_classes)
        param target: (batch_size, num_utterances)
        param mask: (batch_size, num_utterances)
        '''
        mask_ = mask.view(-1,1) 
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)/torch.sum(self.weight[target]*mask_.squeeze())
        return loss

# 该部分代码参考github项目 https://github.com/declare-lab/conv-emotion
class EncoderModel(nn.Module):
    def __init__(self, D_h, cls_model, transformer_model_family, mode, attention=False, residual=False):
        '''
        param transformer_model_family: bert or roberta or xlnet
        param mode: 0(base) or 1(large)
        '''
        super().__init__()
        
        if transformer_model_family == 'bert':
            if mode == '0':
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                hidden_dim = 768
            elif mode == '1':
                model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                hidden_dim = 1024       
        elif transformer_model_family == 'roberta':
            if mode == '0':
                model = RobertaForSequenceClassification.from_pretrained('roberta-base')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                hidden_dim = 768
            elif mode == '1':
                model = RobertaForSequenceClassification.from_pretrained('roberta-large')
                tokenizer = BertTokenizer.from_pretrained('roberta-large')
                hidden_dim = 1024      
        elif transformer_model_family == 'xlnet':
            if mode == '0':
                model = XLNetModel.from_pretrained('xlnet-base-cased')
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                hidden_dim = 768
                
        self.transformer_model_family = transformer_model_family
        self.model = model.cuda()
        self.hidden_dim = hidden_dim
        self.cls_model = cls_model
        self.D_h = D_h
        self.residual = residual
        if transformer_model_family == 'xlnet':
            if mode == '0':
                self.model.mem_len = 900
                self.model.attn_type = 'bi'
        
        if self.transformer_model_family in ['bert', 'roberta', 'xlnet']:
            self.tokenizer = tokenizer
        
    def pad(self, tensor, length):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor
  
    def forward(self, conversations, lengths, umask, qmask):
        '''
        param conversations: 包含语句序列的对话，list
        param lengths: 每个对话的长度，list
        返回值：经过bert编码后的语义向量，及掩码mask
        '''
        # 提取多个对话中的所有句子
        lengths = torch.Tensor(lengths).long()
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent for conv in conversations for sent in conv]
        
        if self.transformer_model_family in ['bert', 'roberta']:
            # 分词
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt") 
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            # 输入bert，返回[CLS]位的向量作为语义向量
            _, features = self.model(input_ids, attention_mask, output_hidden_states=True) 
            if self.transformer_model_family == 'roberta':
                features = features[:, 0, :]
                
        elif self.transformer_model_family == 'xlnet':
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt")
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            features, new_mems = self.model(input_ids, None)[:2]
            features = features[:, -1, :]
        
        # 把输出的features重新组织成batch的形式，(total_utterances_num, hidden_dim) -> (utterances_num_per_conversation, batch_size, hidden_dim)
        features = torch.stack([self.pad(features.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        # 计算mask掩码
        umask = umask.cuda()
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, 2*self.D_h) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        
        return features, umask, mask

class LstmLayer(nn.Module):
    def __init__(self, hidden_dim, D_h):
        super().__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=D_h,num_layers=2, bidirectional=True).cuda() #biLSTM

    def forward(self, features, mask):
        hidden, _ = self.lstm(features)
        hidden = hidden * mask
        return hidden # 返回biLSTM时序编码后的特征

# 该部分代码参考博客 https://blog.csdn.net/qq_44766883/article/details/112008655
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        pe = torch.from_numpy(position_encoding).float()
        pe.cuda()
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        '''
        param x: (utterances_num, batch_size, d_model)
        '''
        x = torch.transpose(x,0,1)
        x = x + self.pe[:, :x.size(1)]
        x = torch.transpose(x,0,1)
        return self.dropout(x)
    
class LinearClassifer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.smax_fc = nn.Linear(input_dim, num_classes).cuda()

    def forward(self, input):
        logits = self.smax_fc(input) # 映射至标签空间
        log_prob = F.log_softmax(logits, 2) # 归一化得到概率向量
        return log_prob, logits # 返回概率向量，和logits
    
class JointModel(nn.Module):
    def __init__(self, D_h, cls_model, transformer_model_family, mode, num_classes, num_subclasses, context_encoder_layer, attention=False, residual=False, no_lstm=False):
        '''
        param D_h: lstm隐藏层大小
        param transformer_model_family: bert or roberta or xlnet
        param mode: 0(base) or 1(large)
        param num_classes: num of emotion classes 
        param num_subclasses: num of emotion bias classes
        param context_encoder_layer: context transformer layer size
        '''
        super().__init__()
        # 语句编码器
        self.encoderModel = EncoderModel(D_h, cls_model, transformer_model_family, mode, attention, residual)
        
        # 情感偏移感知任务
        if mode == '0':
            hidden_dim = 768
        elif mode == '1':
            hidden_dim = 1024
        self.no_lstm = no_lstm
        if no_lstm == False:
            self.lstmLayer = LstmLayer(hidden_dim, D_h)
            self.subClassifer = LinearClassifer(D_h * 2, num_subclasses)
        else:
            self.subClassifer = LinearClassifer(hidden_dim, num_subclasses)
        
        # 语句情感识别任务
        self.positionLayer = PositionalEncoding(d_model=num_subclasses + hidden_dim, dropout=0.1).cuda()
        self.tfLayer = nn.TransformerEncoderLayer(d_model=num_subclasses + hidden_dim, nhead=3).cuda()  # (768+3)/3
        self.tfNorm = nn.LayerNorm(num_subclasses + hidden_dim)
        self.tfEncoder = nn.TransformerEncoder(self.tfLayer, context_encoder_layer, norm=self.tfNorm).cuda()
        self.mainClassifer = LinearClassifer(num_subclasses + hidden_dim, num_classes)

    def find_last(self, utterances, index):
        '''
        param utterances: 语义特征，tensor, (utterances_num_per_conversation, batch_size, hidden_dim)
        param index: 每个句子同一说话者上句的编号，tensor, (utterances_num_per_conversation, batch_size)
        返回值：逐语句的向量相似度
        '''
        ## max_conv_len, bsz, hidden_dim
        utterances = utterances.cuda()
        ## 如果bsz!=1, 则不改变
        utterances = torch.squeeze(utterances, 1)  # assume batch_size = 1
        index = torch.squeeze(index, 1)
        # 对于对话中第一个句子，没有上句的情况，增加全零的向量作为其前句
        # bsz=2 : RuntimeError: Tensors must have same number of dimensions: got 3 and 2
        utterances_ = torch.cat((torch.zeros(1, utterances.size(1)).cuda(), utterances), 0)
        index_ = index + 1
        # 根据传入的index，取出对应的上句向量
        last_utterances = torch.index_select(utterances_, 0, index_)
        last_utterances = torch.unsqueeze(last_utterances, 1)
        utterances = torch.unsqueeze(utterances,1)
        index = torch.unsqueeze(index, 1)
        # 向量作差
        sub = torch.sub(utterances, last_utterances)
        return sub
    '''
    deprecated version
    def find_last(self, utterances, index):
        utterances = utterances.cuda()
        sub = torch.zeros(utterances.size(0), utterances.size(2)).cuda()
        # 遍历每个batch，即每个对话
        for batch in range(utterances.size(1)):
            utt_ = utterances[:, batch, :]  
            utt_ = torch.squeeze(utt_, 1)
            index_ = index[:, batch]
            utterances_ = torch.cat((torch.zeros(1, utt_.size(1)).cuda(), utt_), 0)
            index_ = index_ + 1
            last_utterances = torch.index_select(utt_, 0, index_)
            sub = torch.stack((sub, torch.sub(utt_, last_utterances)), dim = 1)
        return sub[:, 1:, :]
    '''

    def forward(self, conversations, subindex, lengths, umask, qmask):
        '''
        param conversations: 包含语句序列的对话，list
        param lengths: 每个对话的长度，list
        param subindex: 每个句子同一说话者上句的编号，tensor, (utterances_num_per_conversation, batch_size)
        返回值：两个任务的概率向量
        '''
        # 语句编码器
        features, umask, mask = self.encoderModel(conversations, lengths, umask, qmask) # 输入对话中的语句，返回bert编码后的语义向量
        
        # 情感偏移感知任务
        subdata = self.find_last(features, subindex) # 传入subindex，是每个句子同一说话者上句的编号，返回逐语句的向量相似度作为偏移向量
        if self.no_lstm == False:
            lstm_features = self.lstmLayer(subdata, mask) # lstm时序编码
            sub_output, sub_logits = self.subClassifer(lstm_features) # 分类器，返回的sub_output是情感偏移概率，sub_logits是情感偏移特征
        else:
            sub_output, sub_logits = self.subClassifer(subdata)
        
        # 语句情感识别任务
        fushion_features = torch.cat((features, sub_logits.detach()), 2) # 拼接语义特征和情感偏移特征，用.detach()阻止情感偏移特征的梯度回传
        fushion_features = self.positionLayer(fushion_features) # 为语句增加位置编码
        tf_features = self.tfEncoder(fushion_features) # transformer语境编码器
        main_output, _ = self.mainClassifer(tf_features) # 分类器，返回的main_output是情感类别概率
        return main_output, sub_output