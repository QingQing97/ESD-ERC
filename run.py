# import warnings
# warnings.filterwarnings('always')

import numpy as np, random, math
from tqdm import tqdm
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from dataloader import DialogLoader
from model import JointModel, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

import sys

'''
anger	0 
no_emotion	1 
disgust	2 
fear	3 
happiness	4 
sadness	5
surprise	6
'''

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 1022, 1: 85572, 2: 353, 3: 174, 4: 12885, 5: 1150, 6: 1823}
    # the original weights in COSMIC:
    # labels_dict = {0: 12885, 1: 85572, 2: 1022, 3: 1150, 4: 174, 5: 1823, 6: 353}
    # 0 happy, 1 neutral, 2 anger, 3 sad, 4 fear, 5 surprise, 6 disgust 
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params":  ([p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer

def configure_dataloaders(dataset, classify, multitask, batch_size):
    "Prepare dataloaders"
    train_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
    valid_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
    test_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_loss_mask.tsv'

    train_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + multitask + '_label.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_subtask01_index.tsv',
        train_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_speakers.tsv',
        batch_size,
        shuffle=True
    )
    
    valid_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + classify + '.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + multitask + '_label.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_subtask01_index.tsv',
        valid_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_speakers.tsv',
        batch_size,
        shuffle=False
    )
    
    test_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + multitask + '_label.tsv',
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_subtask01_index.tsv',
        test_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_speakers.tsv',
        batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader


def metric_helper(dataset, labels, preds, masks, losses, task_type):
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)

    if dataset in ['iemocap']:
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        fscores = [avg_fscore]

    elif dataset == 'dailydialog':
        if task_type == 'main':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
            avg_fscore2 = round(
                f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0, 2, 3, 4, 5, 6]) * 100,
                2)  # 去除中性类别
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
            avg_fscore4 = round(
                f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0, 2, 3, 4, 5, 6]) * 100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
            avg_fscore6 = round(
                f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0, 2, 3, 4, 5, 6]) * 100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
        elif task_type == 'sub':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
            fscores = [avg_fscore1, avg_fscore3, avg_fscore5]

    elif dataset == 'meld':
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro') * 100, 2)
        avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
        fscores = [avg_fscore1, avg_fscore3, avg_fscore5]

    return avg_loss, avg_accuracy, fscores, labels, preds, masks

def train_or_eval_model(model, main_loss_function, sub_loss_function, dataloader, epoch, acc_steps, optimizer=None, train=False, grad_acc=False):
    losses1, losses2, preds1, preds2, labels1, labels2, masks1, masks2 = [], [], [], [], [], [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()

    i = 0
    # seed_everything(seed)
    # batch size == 1代表1个batch包含1个conversation

    conv_num = 0
    if save_badCases:
        print('Save Bad Cases')
        fcases = open('BadCases_' +'ESD_ERC' + '_' + dataset + '_' + 'fscore' + eval_num + '.txt','w')

    for conversations, label, sublabel, subindex, loss_mask, speaker_mask in tqdm(dataloader, leave=False):

        # 数据处理，计算lengths和utterance_mask，为了让同一个batch内的不同长度的conversation做padding
        # create umask and qmask 
        lengths = [len(item) for item in conversations]
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1

        # meld的speakers是人名，还没处理
        qmask = speaker_mask
        # qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask], batch_first=False).long().cuda()
        # qmask = torch.nn.functional.one_hot(qmask)

        # 默认batch =1，无所谓pad了
        subindex = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in subindex],
                                                   batch_first=False, padding_value=-1).long().cuda()
        # create labels
        label1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label],
                                                batch_first=True).cuda()
        label2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in sublabel],
                                                batch_first=True).cuda()
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                    batch_first=True).long().cuda()

        #送进模型的conversations是不同长度的
#         print('conversations:', conversations)
#         print('lengths:', lengths)
#         print('subindex:', subindex.size())
        # obtain log probabilities
        if train:
            log_prob1, log_prob2 = model(conversations, subindex, lengths, umask, qmask) # log_prob: (utterances_num, batch_size, classes_num)
        else:
            with torch.no_grad():
                log_prob1, log_prob2 = model(conversations, subindex, lengths, umask, qmask)

        lp1_ = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2])
        # 标签
        labels1_ = label1.view(-1)
        # print(labels1_)
        loss1 = main_loss_function(lp1_, labels1_, loss_mask)
        # 预测结果
        pred1_ = torch.argmax(lp1_, 1)
        # print(pred1_)

        if save_badCases:
            for i, label in enumerate(labels1_) :
                # print(label)
                if label != 1:
                    if pred1_[i] != label:
                        # 写入文件
                        ## conv编号，utterance编号，错误的判断结果
                        fcases.write(str(conv_num) + '  ' + str(i) + ' ' +str(int(pred1_[i])) + ' ' + '\n')
        # 从0开始计数
        conv_num += 1

        
        preds1.append(pred1_.data.cpu().numpy())
        labels1.append(labels1_.data.cpu().numpy())
        masks1.append(loss_mask.view(-1).cpu().numpy())
        losses1.append(loss1.item() * masks1[-1].sum())


        lp2_ = log_prob2.transpose(0, 1).contiguous().view(-1, log_prob2.size()[2])
        labels2_ = label2.view(-1)
        loss2 = sub_loss_function(lp2_, labels2_, loss_mask)
        pred2_ = torch.argmax(lp2_, 1)
        preds2.append(pred2_.data.cpu().numpy())
        labels2.append(labels2_.data.cpu().numpy())
        masks2.append(loss_mask.view(-1).cpu().numpy())
        losses2.append(loss2.item() * masks2[-1].sum())


        loss = loss1 + loss2
        if train:
            if grad_acc:
                accumulation_steps = int(acc_steps)
                loss = loss/accumulation_steps
                loss.backward()
                if ((i + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                i = i + 1
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

#     print('main task labels size:', len(labels1))
#     every_emotion = []
#     for emo_i in range(7):
#         every_emotion.append([[],[]])
#     for conv_i in range(len(labels1)):
#         for utt_i in range(len(labels1[conv_i])):
#             every_emotion[labels1[conv_i][utt_i]][0].append(labels2[conv_i][utt_i])
#             every_emotion[labels1[conv_i][utt_i]][1].append(preds2[conv_i][utt_i])

#     for emotion_i in range(len(every_emotion)):
#         true = every_emotion[emotion_i][0]
#         pred = every_emotion[emotion_i][1]
#         avg_fscore1 = round(f1_score(true, pred, average='weighted', labels=[0,1]) * 100, 2)
#         avg_fscore3 = round(f1_score(true, pred, average='micro', labels=[0,1]) * 100, 2)
#         avg_fscore5 = round(f1_score(true, pred, average='macro', labels=[0,1]) * 100, 2)
#         print('\n' + 'sub task result at emotion '+ str(emotion_i) + '\n')
#         x2 = 'test__fscore {}'.format([avg_fscore1, avg_fscore3, avg_fscore5]) + '\n'
#         print(x2)
#         print('\n' + 'classification report at emotion '+ str(emotion_i) + '\n')
#         print(str(classification_report(true, pred, digits=4, labels=[0,1])) + '\n')

    return list(zip(metric_helper(dataset, labels1, preds1, masks1, losses1, 'main'), metric_helper(dataset, labels2, preds2, masks2, losses2, 'sub')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--lr_decay_type', default='none', help="steplr|exlr")
    parser.add_argument('--lr_decay_param', default=0.5, type=float, help="steplr: 0.5|0.1;exlr:0.98|0.99|0.90")
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class_weight', default='none', help='cosmic|sklearn|none')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--model', default='bert', help='which model family bert|roberta|sbert; sbert is sentence transformers')
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--dataset', default='dailydialog', help='which dataset iemocap|dailydialog|meld')
    parser.add_argument('--classify', default='emotion', help='what to classify emotion')
    parser.add_argument('--multitask', default='subtask01', help='subtask01|subtask01Senti|subtask013Senti')
    parser.add_argument('--grad_acc', action='store_true', default=False, help='use grad accumulation')
    parser.add_argument('--acc_steps', default='1', help='1|2|4|8')
    # parser.add_argument('--seed', type=int, default=777, metavar='seed', help='seed')
    parser.add_argument('--describe', default='run_bishe.py')
    parser.add_argument('--context_encoder_layer', type=int, default=6)
    parser.add_argument('--load_model_path', default='414214990281_valid_fscores5.pth', help='load model')
    
    parser.add_argument('--eval_num', default='3', help='1|3|5')
    parser.add_argument('--save_bad', action='store_true', default=False, help='save model')

    args = parser.parse_args()

    print(args)

    global dataset
    global classify
    D_h = 200 # lstm layer
    batch_size = args.batch_size
    n_epochs = args.epochs
    dataset = args.dataset
    classify = args.classify
    transformer_model = args.model
    transformer_mode = args.mode
    multitask = args.multitask
    grad_acc = args.grad_acc
    acc_steps = args.acc_steps
    context_encoder_layer = args.context_encoder_layer

    global eval_num
    eval_num = args.eval_num
    global save_badCase
    save_badCases = args.save_bad

    # torch.use_deterministic_algorithms(True)
    
    if dataset == 'dailydialog':
        print ('Classifying emotion in dailydialog.')
        n_classes  = 7
    elif dataset == 'meld':
        print ('Classifying emotion in meld.')
        n_classes  = 7

    if multitask == 'subtask01' or multitask == 'subtask01Senti':
        n_subclasses = 3
    elif multitask == 'subtask013Senti':
        n_subclasses = 4

    #load model
    model = JointModel(D_h, None, transformer_model, transformer_mode, n_classes, n_subclasses, context_encoder_layer, False, False)
    checkpoint = torch.load('saved_models/' + str(args.load_model_path))
    model.load_state_dict(checkpoint['model'])
    seed = checkpoint['seed']
    epoch = checkpoint['epoch']
    seed_everything(seed)
    print('seed: ' + str(seed))
    print('epoch: ' + str(epoch))

    '''
    anger	0 
    no_emotion	1 
    disgust	2 
    fear	3 
    happiness	4 
    sadness	5
    surprise	6
    '''
    if args.class_weight == 'cosmic':
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:   
            loss_weights = torch.FloatTensor([4, 0.3, 8, 8, 2, 4, 4])
            # counts {0: 1022, 1: 85572, 2: 353, 3: 174, 4: 12885, 5: 1150, 6: 1823}
            # the original weights in COSMIC:
            # loss_weights = torch.FloatTensor([2, 0.3, 4, 4, 8, 4, 8])
            # 0 happy, 1 neutral, 2 anger, 3 sad, 4 fear, 5 surprise, 6 disgust
            # counts {0: 12885, 1: 85572, 2: 1022, 3: 1150, 4: 174, 5: 1823, 6: 353}
            
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'sklearn':
        # see calculate_class_weights.ipynb
        loss_weights = torch.FloatTensor([14.39460442,0.17191705,41.67503035,84.54761905,1.14173735,12.79242236,8.06982211])
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'none':
        main_loss_function = MaskedNLLLoss()
    sub_loss_function = MaskedNLLLoss()

    _, _, test_loader = configure_dataloaders(dataset, classify, multitask, batch_size)

    start_time = time.time()
    print('-----------test-----------')
    test_result = train_or_eval_model(model, main_loss_function, sub_loss_function, test_loader, epoch, acc_steps)

    # main task result
    print('\n' + 'main task result' + '\n')
    x1 = 'Epoch {}'.format(epoch) + '\n' + \
        'test__loss {} test__acc {} test__fscore {}'.format(test_result[0][0], test_result[1][0], test_result[2][0]) + '\n' + \
        'time {}'.format(round(time.time() - start_time, 2))
    print(x1)

    print('\n' + 'classification report' + '\n')
    print(str(classification_report(test_result[3][0], test_result[4][0],sample_weight=test_result[5][0], digits=4)) + '\n')
    print(str(confusion_matrix(test_result[3][0], test_result[4][0],sample_weight=test_result[5][0])) + '\n')

    # sub task result
    print('\n' + 'sub task result' + '\n')
    x2 = 'Epoch {}'.format(epoch) + '\n' + \
         'test__loss {} test__acc {} test__fscore {}'.format(test_result[0][1], test_result[1][1],test_result[2][1]) + '\n' + \
         'time {}'.format(round(time.time() - start_time, 2))

    print(x2)

    print('\n' + 'classification report' + '\n')
    print(str(classification_report(test_result[3][1], test_result[4][1], sample_weight=test_result[5][1], digits=4)) + '\n')
    print(str(confusion_matrix(test_result[3][1], test_result[4][1], sample_weight=test_result[5][1])) + '\n')

