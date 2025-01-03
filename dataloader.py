import pandas as pd
from torch.utils.data import Dataset, DataLoader

class UtteranceDataset(Dataset):

    # content是某对话的[utterance1, utterance2, ...]
    # utterances是[conversation1, conversation2, ...]
    def __init__(self, filename1, filename2, filename3, filename4, filename5, filename6):
        
        utterances, labels, sublabels, subindex, loss_mask, speakers = [], [], [], [], [], []
        
        with open(filename1) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                utterances.append(content)
        
        with open(filename2) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                labels.append([int(l) for l in content])

        with open(filename3) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                sublabels.append([int(l) for l in content])

        with open(filename4) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                subindex.append([int(l) for l in content])

        with open(filename5) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                loss_mask.append([int(l) for l in content])

        with open(filename6) as f:
            for line in f:
                content = line.strip().split('\t')[1:]
                speakers.append(content)

        self.utterances = utterances
        self.labels = labels
        self.sublabels = sublabels
        self.subindex = subindex
        self.loss_mask = loss_mask
        self.speakers = speakers
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        s = self.utterances[index]
        l = self.labels[index]
        sb = self.sublabels[index]
        i = self.subindex[index]
        m = self.loss_mask[index]
        sp = self.speakers[index]
        return s, l, sb, i, m, sp
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def DialogLoader(filename1, filename2, filename3, filename4, filename5, filename6, batch_size, shuffle):
    dataset = UtteranceDataset(filename1, filename2, filename3, filename4, filename5, filename6)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader