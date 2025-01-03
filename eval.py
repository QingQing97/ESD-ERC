# 用于加载保存好的模型，并进行测试
import numpy as np, random
from tqdm import tqdm
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from dataloader_single import DialogLoader
# 在model上是一致的
from model_single import DialogBertTransformer, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import sys
import json

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 过程应该与train_single.py相似，去除模型训练部分
def configure_dataloaders(dataset, classify, batch_size):
    "Prepare dataloaders"
    train_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
    valid_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
    test_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_loss_mask.tsv'
        
    train_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        train_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_speakers.tsv',  
        batch_size,
        shuffle=True
    )
    
    valid_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + classify + '.tsv',
        valid_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_speakers.tsv', 
        batch_size,
        shuffle=False
    )
    
    test_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        test_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_speakers.tsv', 
        batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None
    
    conv_num = 0
    if save_badCases:
        print('Save Bad Cases')
        fcases = open('BadCases_' + classification_model + '_' + dataset + '_' + 'fscore' + eval_num + '.txt','w')
    # label_num = 0
    for conversations, label, loss_mask, speaker_mask in tqdm(dataloader, leave=False):
        lengths = [len(item) for item in conversations]
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1
        
        # train_single.py版本
        # qmask = speaker_mask
        # train_single_rectify.py版本
        if dataset == "dailydialog":
            ## 强制转化为int型
            speaker_mask = np.array(speaker_mask).astype(dtype=int)
            speaker_mask = torch.from_numpy(speaker_mask)
            ## one-hot
            qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask], batch_first=False).long().cuda()
            qmask = torch.nn.functional.one_hot(qmask)
        elif dataset == "meld":
            for index, conv in enumerate(speaker_mask):
                name_num = {}
                speakers = []
                num = 0
                for pos, name in enumerate(conv) :
                    if not name_num.__contains__(name):
                        name_num[name] = num
                        num += 1
                    speakers.append(name_num[name])
                speaker_mask[index]=speakers
            qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask], batch_first=False).long().cuda()
            qmask = torch.nn.functional.one_hot(qmask)
        
        # create labels and mask
        label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], 
                                                batch_first=True).cuda()
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], 
                                                    batch_first=True).long().cuda()
        
        
        # obtain log probabilities
        # 单个conv中每个句子的情感概率分布
        log_prob = model(conversations, lengths, umask, qmask, None)
        # print(log_prob)
        ### print(log_prob.size())
        ### torch.Size([12, 1, 7])
        ### sys.exit()
        
        # compute loss and metrics
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1) 
        loss = loss_function(lp_, labels_, loss_mask)

        # ---------找出类别---------
        pred_ = torch.argmax(lp_, 1) 
        # print(pred_)
        # print(labels_)
        # sys.exit()
        # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1], device='cuda:0')
        # tensor([1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1], device='cuda:0')
        # label_num += labels_.size()[0]
        # 排除中性类别的情况下，找到bad case
        if save_badCases:
            for i, label in enumerate(labels_) :
                # print(label)
                if label != 1:
                    if pred_[i] != label:
                        # 写入文件
                        ## conv编号，utterance编号，错误的判断结果
                        fcases.write(str(conv_num) + '  ' + str(i) + ' ' +str(int(pred_[i])) + ' ' + '\n')
        # 从0开始计数
        conv_num += 1

        # 保存所有conv中每个句子的预测标签、真实标签
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())

        # 保存所有conv的loss
        masks.append(loss_mask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

    # print(label_num)
    # sys.exit()
    # 7740
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    
    if dataset == 'dailydialog':
        if classify == 'emotion':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            ## 去除中性类别
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            ## 去除中性类别
            avg_fscore4 = round(f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            ## 去除中性类别
            avg_fscore6 = round(f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0,2,3,4,5,6])*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
    
    elif dataset == 'meld':
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
        fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
        
    return avg_loss, avg_accuracy, fscores, labels, preds, masks 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--cls_model', default='lstm', help='lstm|dialogrnn|logreg|transformer')
    parser.add_argument('--model', default='bert', help='which model family bert|roberta|sbert|xlnet')
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--dataset', default='dailydialog',help='which dataset meld|dailydialog')
    parser.add_argument('--classify', default='emotion',help='what to classify emotion|act|intent|er|ee')
    parser.add_argument('--cattn', default='general', help='context attention for dialogrnn simple|general|general2')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm model')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')
    parser.add_argument('--seed', type=int, default=6804, metavar='seed', help='seed')
    parser.add_argument('--describe', default='train_single.py')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model')

    parser.add_argument('--num', default='421158404381', help='saved model model')
    parser.add_argument('--eval_num', default='3', help='1|3|5')
    parser.add_argument('--save_bad', action='store_true', default=False, help='save model')
    args = parser.parse_args()
    print(args)

    global dataset
    global classify
    global classification_model

    D_h = 200
    batch_size = args.batch_size
    dataset = args.dataset
    classification_model = args.cls_model
    transformer_model = args.model
    transformer_mode = args.mode
    context_attention = args.cattn
    attention = args.attention
    residual = args.residual
    global seed
    seed = args.seed
    seed_everything(seed)

    model_number = args.num 
    global eval_num
    eval_num = args.eval_num
    global save_badCase
    save_badCases = args.save_bad
    
    classify = args.classify
    if dataset == 'dailydialog':
        if classify == 'emotion':
            print ('Classifying emotion in dailydialog.')
            n_classes  = 7
    elif dataset == 'meld':
        if classify == 'emotion':
            print ('Classifying emotion in meld.')
            n_classes  = 7

    

    if classification_model == 'lstm':
        modelPath = r'saved_models/bcLSTM_' + model_number + '_valid_fscores' + eval_num + '.pth'
        print('building model..')
        model = DialogBertTransformer(D_h, classification_model, transformer_model, transformer_mode, n_classes, context_attention, attention, residual)
        model.load_state_dict(torch.load(modelPath)["model"])
    elif classification_model == 'dialogrnn':
        print('loading model..')
        modelPath = r'saved_models/dialogrnn_' + model_number + '_valid_fscores' + eval_num + '.pth'
        model = torch.load(modelPath)
    print('Model from: {}'.format(modelPath))
    
    
    # model = torch.load(modelPath)

    print('Begin evaluation')
    _, _, test_loader = configure_dataloaders(dataset, classify, batch_size)
    # valid_loss, valid_acc, valid_fscore, _, _, _ = train_or_eval_model(model, MaskedNLLLoss(), valid_loader)
        
    test_loss, test_acc, test_fscore, test_label, test_pred, test_mask  = train_or_eval_model(model, MaskedNLLLoss(),
                                                                                                  test_loader)
    # x = 'valid_loss {} valid_acc {} valid_fscore {}'.format(valid_loss, valid_acc, valid_fscore) + '\n' + \
    #     'test__loss {} test__acc {} test__fscore {}'.format(test_loss, test_acc, test_fscore) + '\n'
    # print (x)

    print('test_fscore: ', test_fscore)

