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
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights


def seed_everything(seed):
    random.seed(seed)                          ##  random.seed()：使用 random() 生成的随机数将会是同一个
    np.random.seed(seed)                       ##  np.random.seed()：每次生成的随机数都相同
    torch.manual_seed(seed)                    ##  为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)               ##  为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed)           ##  为所有的GPU设置种子;
    torch.backends.cudnn.benchmark = False     ##  固定卷积算法
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

## 评价指标计算部分
def metric_helper(dataset, labels, preds, masks, losses, task_type):
    if preds != []:
        ## 预测标签
        preds = np.concatenate(preds)
        ## 真实标签
        labels = np.concatenate(labels)
        ## loss_mask
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    ## 按话语数平均loss
    avg_loss = round(np.sum(losses) / np.sum(masks), 4) 
    ## accuracy_score: https://blog.csdn.net/weixin_41990278/article/details/90758829
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)

    if dataset == 'dailydialog':
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

                                               ## 训练或验证的算法 
def train_or_eval_model(dataset, mode, model, main_loss_function, sub_loss_function, dataloader, epoch, acc_steps, optimizer=None, train=False, grad_acc=False):
    losses1, losses2, preds1, preds2, labels1, labels2, masks1, masks2 = [], [], [], [], [], [], [], []
    assert not train or optimizer!=None        ## 确保train和optimizer都不为None
    
    if train:                                  ## train/eval模式选择
        model.train()
    else:
        model.eval()

    ## 调试
    ## conv_num = 0 
    for conversations, label, sublabel, subindex, loss_mask, speaker_mask in tqdm(dataloader, leave=False):    
        ## print(conversations)

        print(conversations)
        print(label)
        sys.exit()

        '''
        if conv_num > 10:
            break
        conv_num += 1
        '''

        # create umask and qmask 
        ## 每个对话的句子数
        ## bsz 元素是：cov_len  
        lengths = [len(item) for item in conversations]
        ## print(lengths)
        ## utterance-mask：bsz, max_conv_len
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        ## print(umask)
        for j in range(len(lengths)):
            ## [0,lengths[j])
            umask[j][:lengths[j]] = 1
        ## bsz, conv_len 对于MELD，元素是：speaker_name；对于DailyDialog，元素是：说话者编号，只有'0', '1'
        qmask = speaker_mask
        # qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask], batch_first=False).long().cuda()
        # qmask = torch.nn.functional.one_hot(qmask)
        ## print(subindex)
        ## subindex： bsz, conv_len      当前话语的同一说话者的上一话语编号
        ## 转tensor   max_conv_len, bsz  用 -1 pad
        subindex = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in subindex],
                                                   batch_first=False, padding_value=-1).long().cuda()
        ## print(subindex)
        ## print(label)
        ## bsz, conv_len
        ## 转tensor  bsz, max_conv_len   用 0 pad
        label1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label],
                                                batch_first=True).cuda()
        ## print(label1)
        ## bsz, max_conv_len
        label2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in sublabel],
                                                batch_first=True).cuda()
        ## bsz, max_conv_len  --> 计算loss时的mask
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                    batch_first=True).long().cuda()
        # obtain log probabilities
        if train:
            ## print(log_prob1.shape)
            ## torch.Size([6, 1, 7])
            ## max_utt_num, bsz, label_num
            log_prob1, log_prob2 = model(conversations, subindex, lengths, umask, qmask)
        else:
            with torch.no_grad():
                log_prob1, log_prob2 = model(conversations, subindex, lengths, umask, qmask)
        
        lp1_ = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2]) ## utt_num, label_num

        labels1_ = label1.view(-1)                                ## utt_num 真实标签
        loss1 = main_loss_function(lp1_, labels1_, loss_mask)     ## 1 针对单个对话的整体loss

        pred1_ = torch.argmax(lp1_, 1)                            ## utt_num 预测标签list
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

        loss = loss1 + loss2                ## 联合损失函数

        if train:                           ## 反向传播
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
    ## 主辅任务的评价指标拼一起了
    return list(zip(metric_helper(dataset, labels1, preds1, masks1, losses1, 'main'), metric_helper(dataset, labels2, preds2, masks2, losses2, 'sub')))
                 

def result_helper(valid_fscores, test_fscores, valid_losses, rf, lf, best_label, best_pred, best_mask, task_type):
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    print('Test performance.')
    if dataset == 'dailydialog':
        if task_type == 'main':
            ## 各评价指标相对应的目标值
            score1 = test_fscores[0][np.argmin(valid_losses)]
            score2 = test_fscores[0][np.argmax(valid_fscores[0])]
            score3 = test_fscores[1][np.argmin(valid_losses)]
            score4 = test_fscores[1][np.argmax(valid_fscores[1])]
            score5 = test_fscores[2][np.argmin(valid_losses)]
            score6 = test_fscores[2][np.argmax(valid_fscores[2])]
            score7 = test_fscores[3][np.argmin(valid_losses)]
            score8 = test_fscores[3][np.argmax(valid_fscores[3])]
            score9 = test_fscores[4][np.argmin(valid_losses)]
            score10 = test_fscores[4][np.argmax(valid_fscores[4])]
            score11 = test_fscores[5][np.argmin(valid_losses)]
            score12 = test_fscores[5][np.argmax(valid_fscores[5])]

            ## scores = [score1, score2, score3, score4, score5, score6,
            ##          score7, score8, score9, score10, score11, score12]
            scores_val_loss = [score1, score3, score5, score7, score9, score11]
            scores_val_f1 = [score2, score4, score6, score8, score10, score12]
            ## 最小loss对应的epoch
            loss_at_epoch = np.argmin(valid_losses)
            ## 最大fscores值对应的epoch
            f1_at_epoch = [np.argmax(valid_fscores[0]), np.argmax(valid_fscores[1]), np.argmax(valid_fscores[2]), np.argmax(valid_fscores[3]), \
                           np.argmax(valid_fscores[4]), np.argmax(valid_fscores[5])]

            res1 = 'Scores: Weighted, Weighted w/o Neutral, Micro, Micro w/o Neutral, Macro, Macro w/o Neutral'
            res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
            res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
            res4 = 'loss at epoch:' + str(loss_at_epoch)
            res5 = 'F1 at epoch: {}'.format(f1_at_epoch)

        elif task_type == 'sub':
            score1 = test_fscores[0][np.argmin(valid_losses)]
            score2 = test_fscores[0][np.argmax(valid_fscores[0])]
            score3 = test_fscores[1][np.argmin(valid_losses)]
            score4 = test_fscores[1][np.argmax(valid_fscores[1])]
            score5 = test_fscores[2][np.argmin(valid_losses)]
            score6 = test_fscores[2][np.argmax(valid_fscores[2])]

            ## scores = [score1, score2, score3, score4, score5, score6]
            scores_val_loss = [score1, score3, score5]
            scores_val_f1 = [score2, score4, score6]
            loss_at_epoch = np.argmin(valid_losses)
            f1_at_epoch = [np.argmax(valid_fscores[0]), np.argmax(valid_fscores[1]), np.argmax(valid_fscores[2])]

            res1 = 'Scores: Weighted, Micro, Macro'
            res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
            res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
            res4 = 'loss at epoch:' + str(loss_at_epoch)
            res5 = 'F1 at epoch: {}'.format(f1_at_epoch)

        print(res1)
        print(res2)
        print(res3)
        print(res4)
        print(res5)
        rf.write(res1 + '\n')
        rf.write(res2 + '\n')
        rf.write(res3 + '\n')
        rf.write(res4 + '\n')
        rf.write(res5 + '\n')

        lf.write(res1 + '\n')
        lf.write(res2 + '\n')
        lf.write(res3 + '\n')
        lf.write(res4 + '\n')
        lf.write(res5 + '\n')

        if task_type == 'main':
            rf.write('\n' + 'classification report at best Macro F1 w/o Neutral' + '\n')
            rf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[5])], best_pred[np.argmax(valid_fscores[5])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[5])], digits=4)) + '\n')
            rf.write(
                str(confusion_matrix(best_label[np.argmax(valid_fscores[5])], best_pred[np.argmax(valid_fscores[5])],
                                     sample_weight=best_mask[np.argmax(valid_fscores[5])])) + '\n')
            rf.write('\n' +'classification report at best Macro F1 with Neutral'+ '\n')
            rf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[4])], best_pred[np.argmax(valid_fscores[4])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[4])], digits=4)) + '\n')
            rf.write(str(confusion_matrix(best_label[np.argmax(valid_fscores[4])], best_pred[np.argmax(valid_fscores[4])],
                                          sample_weight=best_mask[np.argmax(valid_fscores[4])])) + '\n')
            rf.write('over'+ '\n\n')
            #
            lf.write('\n' + 'classification report at best Macro F1 w/o Neutral' + '\n')
            lf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[5])], best_pred[np.argmax(valid_fscores[5])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[5])], digits=4)) + '\n')
            lf.write(
                str(confusion_matrix(best_label[np.argmax(valid_fscores[5])], best_pred[np.argmax(valid_fscores[5])],
                                     sample_weight=best_mask[np.argmax(valid_fscores[5])])) + '\n')
            lf.write('\n' + 'classification report at best Macro F1 with Neutral' + '\n')
            lf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[4])], best_pred[np.argmax(valid_fscores[4])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[4])], digits=4)) + '\n')
            lf.write(
                str(confusion_matrix(best_label[np.argmax(valid_fscores[4])], best_pred[np.argmax(valid_fscores[4])],
                                     sample_weight=best_mask[np.argmax(valid_fscores[4])])) + '\n')
            lf.write('over'+ '\n\n')
    
    ## 最优结果输出
    elif dataset == 'meld':
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]

        ## scores = [score1, score2, score3, score4, score5, score6]
        scores_val_loss = [score1, score3, score5]
        scores_val_f1 = [score2, score4, score6]
        loss_at_epoch = np.argmin(valid_losses)                  ## 用数组保存所有结果，返回最小值编号
        f1_at_epoch = [np.argmax(valid_fscores[0]), np.argmax(valid_fscores[1]), np.argmax(valid_fscores[2])]

        res1 = 'Scores: Weighted, Micro, Macro'
        res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
        res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
        res4 = 'loss at epoch:' + str(loss_at_epoch)
        res5 = 'F1 at epoch: {}'.format(f1_at_epoch)
            
        print(res1)
        print(res2)
        print(res3)
        print(res4)
        print(res5)
        rf.write(res1 + '\n')
        rf.write(res2 + '\n')
        rf.write(res3 + '\n')
        rf.write(res4 + '\n')
        rf.write(res5 + '\n')

        lf.write(res1 + '\n')
        lf.write(res2 + '\n')
        lf.write(res3 + '\n')
        lf.write(res4 + '\n')
        lf.write(res5 + '\n')
        
        ## 将主任务结果写入文件
        if task_type == 'main':
            rf.write('\n' + 'classification report at best Macro F1' + '\n')
            rf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[2])], best_pred[np.argmax(valid_fscores[2])],
                                  sample_weight=best_mask[np.argmax(valid_fscores[2])], digits=4))  + '\n')
            rf.write(
                str(confusion_matrix(best_label[np.argmax(valid_fscores[2])], best_pred[np.argmax(valid_fscores[2])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[2])])) + '\n')
            rf.write('over'+ '\n\n')
            #
            lf.write('\n' + 'classification report at best Macro F1' + '\n')
            lf.write(str(
                classification_report(best_label[np.argmax(valid_fscores[2])], best_pred[np.argmax(valid_fscores[2])],
                                  sample_weight=best_mask[np.argmax(valid_fscores[2])], digits=4))  + '\n')
            lf.write(
                str(confusion_matrix(best_label[np.argmax(valid_fscores[2])], best_pred[np.argmax(valid_fscores[2])],
                                      sample_weight=best_mask[np.argmax(valid_fscores[2])])) + '\n')
            
            lf.write('over'+ '\n\n')
    # rf.write(str(args) + '\n')

    rf.close()
    lf.close()
    lf.close()

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
    parser.add_argument('--model', default='bert', help='which model family bert|roberta|xlnet')
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--dataset', default='meld')
    parser.add_argument('--multitask', default='subtask01', help='subtask01|subtask01Senti|subtask013Senti')
    parser.add_argument('--grad_acc', action='store_true', default=False, help='use grad accumulation')
    parser.add_argument('--acc_steps', default='1', help='1|2|4|8')
    parser.add_argument('--seed', type=int, default=777, metavar='seed', help='seed')
    parser.add_argument('--describe', default='train.py')
    parser.add_argument('--context_encoder_layer', type=int, default=6)
    parser.add_argument('--save_model', action='store_true', default=False, help='save model')
    parser.add_argument('--no_lstm', action='store_true', default=False, help='no lstm')
    args = parser.parse_args()

    print(args)

    global dataset                        ## global: 若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global
    D_h = 200 # lstm layer
    batch_size = args.batch_size
    n_epochs = args.epochs
    dataset = args.dataset
    classification_model = 'EBERC'
    transformer_model = args.model
    transformer_mode = args.mode
    multitask = args.multitask
    grad_acc = args.grad_acc
    acc_steps = args.acc_steps
    no_lstm = args.no_lstm
    context_encoder_layer = args.context_encoder_layer
    global seed
    seed = args.seed
    seed_everything(seed)                 ## seed_everything: 自定义函数; seed default=777
    
    if dataset == 'dailydialog':
        print ('Classifying emotion in dailydialog.')
        n_classes  = 7
    elif dataset == 'meld':
        print ('Classifying emotion in meld.')
        n_classes  = 7
        
    if multitask == 'subtask01' or multitask == 'subtask01Senti':  ## multitask default='subtask01' 情感偏移三分类
        n_subclasses = 3
    elif multitask == 'subtask013Senti':
        n_subclasses = 4

    ## from model import JointModel    

    ## transformer_model = args.model = Bert
    model = JointModel(D_h, classification_model, transformer_model, transformer_mode, n_classes, n_subclasses, context_encoder_layer, False, False, no_lstm)

    '''
    anger	0 
    no_emotion	1 
    disgust	2 
    fear	3 
    happiness	4 
    sadness	5
    surprise	6
    '''
    ## 参考了cosmic的损失函数给每个类的权重
    if args.class_weight == 'cosmic':                   ##  default='none'
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:   
            loss_weights = torch.FloatTensor([4, 0.3, 8, 8, 2, 4, 4])
            
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'sklearn':
        # see calculate_class_weights.ipynb
        loss_weights = torch.FloatTensor([14.39460442,0.17191705,41.67503035,84.54761905,1.14173735,12.79242236,8.06982211])
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'none':                                        ## 损失函数
        main_loss_function = MaskedNLLLoss()                                 ## from model import MaskedNLLLoss()
    sub_loss_function = MaskedNLLLoss()

    optimizer = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon)     ## 优化器
    if args.lr_decay_type == 'none':
        pass
    elif args.lr_decay_type == 'exlr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.lr_decay_param)
    elif args.lr_decay_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay_param)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, 'emotion', multitask, batch_size)
    
    lf_main = open('logs/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
              + '_' + multitask + '_main.txt', 'a')
    rf_main = open('results/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
              + '_' + multitask + '_main.txt', 'a')

    lf_sub = open('logs/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
                   + '_' + multitask + '_sub.txt', 'a')
    rf_sub = open('results/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
                   + '_' + multitask + '_sub.txt', 'a')

    ## print('seed: ', seed)                                                          ## 输出seed
    rf_main.write('\n' + str(args) + '\n')
    rf_sub.write('\n' + str(args) + '\n')
    lf_main.write('\n' + str(args) + '\n')
    lf_sub.write('\n' + str(args) + '\n')

    valid_losses1, valid_losses2, valid_fscores1, valid_fscores2 = [], [], [], []
    test_fscores1, test_fscores2 = [], []
    best_loss1, best_loss2 = None, None
    best_label1, best_label2, best_pred1, best_pred2, best_mask1, best_mask2 = [], [], [], [], [], []
    train_losses1, train_losses2, test_losses1, test_losses2 = [], [], [], []

    # if args.save_model:
    saved_model_number = int(time.time() * 256)                                    ## 随机生成保存模型的编号
    ## print('saved_model_number is: ' + str(saved_model_number))
    rf_main.write('saved model number is: ' + str(saved_model_number) + '\n')
    rf_sub.write('saved model number is: ' + str(saved_model_number) + '\n')
    lf_main.write('saved model number is: ' + str(saved_model_number) + '\n')
    lf_sub.write('saved model number is: ' + str(saved_model_number) + '\n')

    for e in range(n_epochs):
        start_time = time.time()                                               ## 记录程序开始的时间                   
        print('---------train--------')                                        
        train_result = train_or_eval_model(dataset, 0, model, main_loss_function, sub_loss_function, train_loader, e, acc_steps, optimizer, True, grad_acc)
        print('-----------valid-----------')                                   
        valid_result = train_or_eval_model(dataset, 1, model, main_loss_function, sub_loss_function, valid_loader, e, acc_steps)
        print('-----------test-----------')
        test_result = train_or_eval_model(dataset, 2, model, main_loss_function, sub_loss_function, test_loader, e, acc_steps)

        if args.lr_decay_type != 'none':
            print("第%d个epoch的学习率：%f" % (e, optimizer.param_groups[0]['lr']))
            scheduler.step()

        # main task result
        ## avg_loss, avg_accuracy, fscores, labels, preds, masks
        valid_losses1.append(valid_result[0][0])
        valid_fscores1.append(valid_result[2][0])
        test_losses1.append(test_result[0][0])
        test_fscores1.append(test_result[2][0])
        train_losses1.append(train_result[0][0])
        
        ## bset vaild loss
        if best_loss1 == None or best_loss1 > valid_result[0][0]:             ## 更新最优loss
            best_loss1 = valid_result[0][0]
        best_label1.append(test_result[3][0])
        best_pred1.append(test_result[4][0])
        best_mask1.append(test_result[5][0])

        ## print(train_result[2][0])
        ## meld: [32.42, 47.02, 11.06]  
        ## dailydialog: [81.32, 32.67, 84.66, 35.91, 19.38, 7.37]        
        x1 = 'Epoch {}'.format(e) + '\n' + 'train_loss {} train_acc {} train_fscore {}'.format(train_result[0][0], train_result[1][0], train_result[2][0]) + '\n' + \
            'valid_loss {} valid_acc {} valid_fscore {}'.format(valid_result[0][0], valid_result[1][0], valid_result[2][0]) + '\n' + \
            'test__loss {} test__acc {} test__fscore {}'.format(test_result[0][0], test_result[1][0], test_result[2][0]) + '\n' + \
            'time {}'.format(round(time.time() - start_time, 2))
        print(x1)
        lf_main.write(x1 + '\n')

        # sub task result
        ## avg_loss, avg_accuracy, fscores, labels, preds, masks
        valid_losses2.append(valid_result[0][1])
        valid_fscores2.append(valid_result[2][1])
        test_losses2.append(test_result[0][1])
        test_fscores2.append(test_result[2][1])
        train_losses2.append(train_result[0][1])

        if best_loss2 == None or best_loss2 > valid_result[0][1]:             ## 更新最优loss
            best_loss2 = valid_result[0][1]
        best_label2.append(test_result[3][1])
        best_pred2.append(test_result[4][1])
        best_mask2.append(test_result[5][1])

        x2 = 'Epoch {}'.format(e) + '\n' + 'train_loss {} train_acc {} train_fscore {}'.format(train_result[0][1],train_result[1][1],train_result[2][1]) + '\n' + \
             'valid_loss {} valid_acc {} valid_fscore {}'.format(valid_result[0][1], valid_result[1][1], valid_result[2][1]) + '\n' + \
             'test__loss {} test__acc {} test__fscore {}'.format(test_result[0][1], test_result[1][1],test_result[2][1]) + '\n' + \
             'time {}'.format(round(time.time() - start_time, 2))

        print(x2)
        lf_sub.write(x2 + '\n')

        # save model
        if args.save_model:         
            fscores_ = np.array(valid_fscores1).transpose()
            state = {'model': model.state_dict(), 'epoch': e, 'seed': seed}
            
            if dataset == 'dailydialog':
                for i in [1, 3, 5]:
                    if np.argmax(fscores_[i]) == np.size(fscores_, 1) - 1:
                        ## print('---------save best fscore model--------')
                        torch.save(state, 'saved_models/' + str(saved_model_number) + '_valid_fscores' + str(i) + '.pth')
            elif dataset == 'meld':
                for i in [0, 1, 2]:
                    if np.argmax(fscores_[i]) == np.size(fscores_, 1) - 1:
                        ## print('---------save best fscore model--------')
                        torch.save(state, 'saved_models/' + str(saved_model_number) + '_valid_fscores' + str(i) + '.pth')
   
    print('saved_model_number is: ' + str(saved_model_number))
    print('seed: ', seed) 
   
    ## 主任务结果
    result_helper(valid_fscores1, test_fscores1, valid_losses1, rf_main, lf_main, best_label1, best_pred1, best_mask1, 'main')
    ## 辅助任务结果
    result_helper(valid_fscores2, test_fscores2, valid_losses2, rf_sub, lf_sub, best_label2, best_pred2, best_mask2, 'sub')


