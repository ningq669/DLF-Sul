import torch
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
import time
import pandas as pd
def encode(string):
    n = len(string)
    m_l = len(string[0])
    x_BE = np.zeros((n, m_l, 20))
    x_62 = np.zeros((n, m_l, 20))
    x_AAI = np.zeros((n, m_l, 14))
    maxabs = preprocessing.MinMaxScaler()
    for i in range(n):
        x_BE[i], x_62[i], x_AAI[i] = forward(string[i])
    x_BE = x_BE.reshape(-1, 1)
    x_62 = x_62.reshape(-1, 1)
    x_AAI = x_AAI.reshape(-1, 1)
    x_BE = maxabs.fit_transform(x_BE)
    x_62 = maxabs.fit_transform(x_62)
    x_AAI = maxabs.fit_transform(x_AAI)
    x_BE = x_BE.reshape(n, m_l, -1)
    x_62 = x_62.reshape(n, m_l, -1)
    x_AAI = x_AAI.reshape(n, m_l, -1)
    x_BE = torch.tensor(x_BE, dtype=torch.float32, requires_grad=True)
    x_62 = torch.tensor(x_62, dtype=torch.float32, requires_grad=True)
    x_AAI = torch.tensor(x_AAI, dtype=torch.float32, requires_grad=True)
    input = torch.cat((x_BE, x_62, x_AAI), 2)
    return input

def forward(x):
    x_BE = BE(x)
    x_62 = BLOSUM62(x)
    x_AAI = AAI(x)
    return x_BE, x_62, x_AAI


def BE(gene):
    with open("BE.txt") as f:
        records = f.readlines()[1:]
    BE = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        BE.append(array)
    BE = np.array(
        [float(BE[i][j]) for i in range(len(BE)) for j in range(len(BE[i]))]).reshape((20, 20))
    BE = BE.transpose()
    AA = 'ACDEFGHIKLMNPQRSTWYV'
    GENE_BE = {}
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = BE[(GENE_BE[gene[i]])]
    return gene_array


def BLOSUM62(gene):
    with open("blosum62.txt") as f:
        records = f.readlines()[1:]
    blosum62 = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        blosum62.append(array)
    blosum62 = np.array(
        [float(blosum62[i][j]) for i in range(len(blosum62)) for j in range(len(blosum62[i]))]).reshape((20, 20))
    blosum62 = blosum62.transpose()
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYV'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = blosum62[(GENE_BE[gene[i]])]
    return gene_array


def AAI(gene):
    with open("AAI.txt") as f:
        records = f.readlines()[1:]
    AAI = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AAI.append(array)
    AAI = np.array(
        [float(AAI[i][j]) for i in range(len(AAI)) for j in range(len(AAI[i]))]).reshape((14, 21))
    AAI = AAI.transpose()
    GENE_BE = {}
    AA = 'ACDEFGHIKLMNPQRSTWYV*'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 14))
    for i in range(n):
        gene_array[i] = AAI[(GENE_BE[gene[i]])]
    return gene_array


def test_data():
    dataset = pd.read_csv(r'test.csv')
    information = dataset.iloc[:, 0:2].values
    test_string = dataset.iloc[:, 2].values
    input = encode(test_string)
    return torch.FloatTensor(input),np.array(information)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blstm = nn.LSTM(input_size=54, hidden_size=n_hidden, bidirectional=True, batch_first=True)
        self.W_Q = nn.Linear(n_hidden * 2, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(n_hidden * 2, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(n_hidden * 2, d_v * n_heads, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(n_heads * d_v, n_hidden * 2, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=6, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.FC = nn.Sequential(
            nn.Linear(3224, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
            torch.nn.Sigmoid()
        )

    def attention(self, input_Q, input_K, input_V, d_model):
        residual, batch_size1 = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size1, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size1, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size1, -1, n_heads, d_v).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(batch_size1, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + residual)

    def forward(self, X):
        X = X.unsqueeze(dim=0)
        batch_size = X.shape[0]
        hidden_state = torch.zeros(1 * 2, batch_size, n_hidden)
        cell_state = torch.zeros(1 * 2, batch_size, n_hidden)
        outputs, (_, _) = self.blstm(X, (hidden_state, cell_state))
        d_model = outputs.size()[-1]
        enc_inputs = outputs
        x_attention_outputs = self.attention(enc_inputs, enc_inputs, enc_inputs, d_model)
        x_CNN_in = x_attention_outputs.unsqueeze(1)
        outputs = self.conv(x_CNN_in)
        outputs = outputs.view(batch_size, -1)
        model = self.FC(outputs)
        return model


def test(test_batch_p):
    model.eval()
    outputs_p_list = []
    n = test_batch_p.size(0)
    s_time = time.time()
    for i in range(n):
        output = model(test_batch_p[i])
        outputs_p_list.append(output)
        e_time = time.time()
        during_time = e_time - s_time
        average_time = during_time / (i + 1)
        remaining_time = average_time * (n - (i + 1))
        print("\r predicting: %.02f%%, remaining: %d seconds" % ((i + 1) / n * 100, remaining_time), end="")
    outputs = torch.cat(outputs_p_list,0)
    outputs = outputs.detach().numpy()
    return outputs[:, 0]


if __name__ == '__main__':
    n_hidden = 64
    d_k = d_v = 64
    n_heads = 8
    test_batch,information = test_data()
    model = Network()
    model.load_state_dict(torch.load("best.mdl"))
    outputs = test(test_batch)
    outputs = outputs.reshape(-1,1)
    output = np.concatenate((information, outputs), axis=1)
    np.savetxt("probability_of_Sul.txt", output, fmt='%s %d %f', delimiter='\n')

