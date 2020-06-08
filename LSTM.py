import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as trns
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

import os
import sys
import random
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from PIL import Image
from IPython import display
from tqdm import tqdm, trange
from scipy.signal import find_peaks, resample
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# cut beat to sequence
train_beats, train_label = [], []
test_beats, test_label = [], []
s_beats, s_label = [], []
e_beats, e_label = [], []

pbar = trange(45)


def to_sequence(target=0, len_thres=128, peak_thres=0.5):
    heart = np.loadtxt('exercise_ECGID/%d.txt' % target, delimiter=' ')
    peaks, height = find_peaks(heart, height=peak_thres)
    data = []
    for k in range(len(peaks) - 1):
        if peaks[k + 1] - peaks[k] > len_thres:
            data.append(heart[peaks[k]:peaks[k + 1]])
    return data


for i in pbar:
    stationary = to_sequence(target=2 * i + 1)
    exercise = to_sequence(target=2 * i + 2)
    s_train, s_test = train_test_split(stationary, test_size=0.3)
    e_train, e_test = train_test_split(exercise, test_size=0.3)

    train_beats += s_train + e_train
    test_beats += s_test + e_test

    s_beats += s_train + s_test
    e_beats += e_train + e_test

    train_label += [i for _ in range(len(s_train) + len(e_train))]
    test_label += [i for _ in range(len(s_test) + len(e_test))]

    s_label += [i for _ in range(len(s_train) + len(s_test))]
    e_label += [i for _ in range(len(e_train) + len(e_test))]

    pbar.set_description("beats: %d" % (len(train_beats) + len(test_beats)))

print("train_beats %d" % len(train_beats))
print("train_label %d" % len(train_label))
print("test_beats %d" % len(test_beats))
print("test_label %d" % len(test_label))

print("station_beats %d" % len(s_beats))
print("station_label %d" % len(s_label))
print("exercise_beats %d" % len(e_beats))
print("exercise_label %d" % len(e_label))


# dataset & dataloader
class BeatDataset(Dataset):
    def __init__(self, datas, labels):
        self.beats = datas
        self.labels = labels

    def __getitem__(self, index):
        beat = self.beats[index]

        # (-1,1)
        # beat = (beat-np.min(beat))/(np.max(beat)-np.min(beat))*2-1

        # standardize
        beat = (beat - np.mean(beat)) / np.std(beat)

        beat = torch.Tensor(np.expand_dims(beat, axis=1))
        downsample = resample(beat, 128)
        downsample = np.squeeze(downsample, axis=1)
        label = self.labels[index]

        return beat, label, downsample

    def __len__(self):
        return len(self.labels)


batch_size = 16
split_ratio = 0.8

train_set = BeatDataset(train_beats, train_label)
test_set = BeatDataset(test_beats, test_label)
rest_set = BeatDataset(s_beats, s_label)
exercise_set = BeatDataset(e_beats, e_label)


# rnn_utils.pack_sequence
def collate_fn(batch):
    data = [item[0] for item in batch]
    downsample = torch.Tensor([item[2] for item in batch])
    lengths = torch.Tensor([len(item[0]) for item in batch])
    label = torch.Tensor([item[1] for item in batch])
    data = rnn_utils.pad_sequence(data, batch_first=False, padding_value=0)
    data = rnn_utils.pack_padded_sequence(data, lengths, batch_first=False, enforce_sorted=False)
    return data, label, downsample


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

rest_loader = DataLoader(rest_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
exercise_loader = DataLoader(exercise_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# for bs, ls, ds in train_loader:
#     print(ls.shape)
# #     test = torch.squeeze(test)
# #     print(test.data.shape)
#     break


class LSTMmodule(nn.Module):
    def __init__(self):
        super(LSTMmodule, self).__init__()
        self.input_feature_dim = 1
        self.hidden_feature_dim = 128
        self.hidden_layer_num = 1
        self.latent_size = 45   # 32
        self.de_hidden_layer_num = 1
        self.de_hidden_feature_dim = 1
        # ---------------------------------------------------------------------------------------
        self.lstm = nn.LSTM(self.input_feature_dim, self.hidden_feature_dim, self.hidden_layer_num)
        self.linear1 = nn.Linear(self.hidden_feature_dim, self.latent_size)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        hn = torch.squeeze(hn)
        x = self.linear1(hn)

        return x


learning_rate = 0.001
l = LSTMmodule()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
l.to(device)
optimizer = optim.Adam(l.parameters(), lr=learning_rate)

loss_log = []
losses = []

epoch = 50
t = trange(epoch)

for e in t:
    for bs, ls, ds in train_loader:
        bs = bs.to(device)
        ls = ls.to(device)

        optimizer.zero_grad()
        x = l.forward(bs)
        loss = F.cross_entropy(x, ls.long())
        loss.backward()
        optimizer.step()

        t.set_description("loss: %f" % loss.item())
        losses.append(loss.item())

    loss_log.append(np.mean(losses))
    losses = []

plt.figure()
plt.plot(loss_log)
plt.title("train loss")
plt.show()
l.eval()

loss_log = []
losses = []
pred = []
test_label = []

for bs, ls, ds in test_loader:
    bs = bs.to(device)
    ls = ls.to(device)
    x = l.forward(bs)

    p = torch.argmax(x, dim=1)
    pred.append(p.cpu().detach().numpy())
    test_label.append(ls.long().cpu().detach().numpy())


pred = np.array(pred).squeeze()
test_label = np.array(test_label).squeeze()

# pred_one = OneHotEncoder().fit_transform(pred)
# test_label_one = OneHotEncoder().fit_transform(test_label)
# cf_mat = confusion_matrix(test_label_one, pred_one, labels=list(range(1, 46)))

# accu = np.where(test_label == pred)
accu = 0
total = 0
for i in range(len(pred)):
    for j in range(pred[i].shape[0]):
        total = total + 1
        if pred[i][j] == test_label[i][j]:
            accu = accu + 1
print("accuracy: ", accu/total)

# plt.figure(figsize=(8, 6))
# plt.title("LSTM")

# # sn.heatmap(cf_mat, annot=True, fmt="d", cmap="YlGnBu")
# sn.heatmap(cf_mat, cmap="YlGnBu")

#b, t = plt.ylim()
# b += 1
# t -= 1
# plt.ylim(b, t)
# plt.show()
