# -*- coding: utf-8 -*-
"""q1_rnn_novel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bUX7ZnwcKEMhacA9QL1GqxzaP3cFTqjM
"""

from google.colab import drive
drive.mount('/content/drive/')

cd drive/My Drive/DL2/

cd DL2

import torch   

#handling text data
from torchtext import data  
import torch.optim as optim

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
LABEL = data.LabelField()
# fields = [('label', LABEL), ('text',TEXT)]
fields = [(None, None), ('label', LABEL), ('text', TEXT)]
train_data=data.TabularDataset(path = 'p_training_data.csv',fields = fields, format = 'csv',skip_header = True)
valid_data = data.TabularDataset(path = 'p_validation_data.csv',fields = fields, format = 'csv',skip_header = True)
#print preprocessed text
print(vars(train_data.examples[0]))

SEED = 2019

#Torch
torch.manual_seed(SEED)

#Cuda algorithms
torch.backends.cudnn.deterministic = True

# import random
# train_data, valid_data = training_data.split(split_ratio=0.8, random_state = random.seed(SEED))

vars(train_data[0])

TEXT.build_vocab(train_data,min_freq= 3,vectors = "glove.840B.300d")  
LABEL.build_vocab(train_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(LABEL.vocab.freqs.most_common(13))  

#Word dictionary
# print(TEXT.vocab.stoi)  
word_embeddings = TEXT.vocab.vectors 
vocab_size = len(TEXT.vocab)

print(LABEL.vocab.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

#set batch size
BATCH_SIZE = 32

#Load an iterator
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)

import torch.nn as nn
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad = False)
        
        #lstm layer
        self.lstm = nn.RNN(64, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        self.conv1 = nn.Conv1d(300, 256, kernel_size = 1)
        self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = 1)
        self.conv2 = nn.Conv1d(256, 128 , kernel_size = 2)
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size = 1)
        self.pool3 = nn.MaxPool1d(kernel_size = 2, stride = 3)

        #dense layer
        self.fc = nn.Linear(4 * hidden_dim, 13)
        
        #activation function
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, text, text_lengths):
        # print("Here")
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        # print(embedded.size())
        # embedded = [batch size, sent_len, emb dim]
        # print("here2")
        embedded = embedded.permute(0, 2, 1)
        # print(embedded.size())
        cnn1 = self.conv1(embedded)
        cnn1 = self.dropout(F.relu(cnn1))
        # print(cnn1.size())
        cnn1_p = self.pool1(cnn1)
        # print(cnn1_p.size())
        cnn2 = self.conv2(cnn1_p)
        cnn2 = self.dropout(F.relu(cnn2))
        # cnn2_sk1 = torch.cat((cnn2, cnn1), dim = 1)
        cnn2_p = self.pool2(cnn2)
        
        # print(cnn2_sk1.size())
        cnn3 = self.conv3(cnn2_p)
        cnn3 = self.dropout(F.relu(cnn3))
        cnn3_p = self.pool3(cnn3)
        cnn3_p = cnn3_p.permute(0, 2, 1)
        
        packed_output, h_n = self.lstm(cnn3_p)
        h_n = self.dropout(h_n)
        h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_size)
        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
        outputs=self.fc(h_n)
        # print(dense_outputs.size())
        return outputs

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train(model, iterator, optimizer, criterion, eps):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    model.cuda()
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text, text_lengths = batch.text  
        # if (text.size()[0] is not 32):
        #   # print("Here")
        #   # One of the batch returned by BucketIterator has length different than 32.
        #   continue 
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
          text = text.cuda()
          target = target.cuda()

        #convert to 1D tensor
        # predictions = model(text, text_lengths).squeeze()  
        
        predictions = model(text, text_lengths).squeeze()
        # print(predictions.size())
        #compute the loss
        loss = criterion(predictions, batch.label) 
        # print(loss.item())       
        # print(target.size())
        #compute the binary accuracy
        # print(predictions.size())
        acc = (torch.max(predictions, 1)[1].view(target.size()).data == target.data).float().sum()/len(batch)
        if eps == 398:
          print(torch.max(predictions, 1)[1].view(target.size()).data, target.data, "training")
        #backpropage the loss and compute the gradients
        # print(torch.max(predictions, 1)[1].view(target.size()).data, "Predict", target.data)
        # print(torch.min(predictions, 1)[1].view(target.size()).data, "Min")
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #deactivating dropout layers
    model.eval()
    model.cuda()
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #retrieve text and no. of words
            text, text_lengths = batch.text
            
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
              text = text.cuda()
              target = target.cuda()
            
            #convert to 1d tensor
            # predictions = model(text, text_lengths).squeeze()
            predictions = model(text, text.size()[0]).squeeze()
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = (torch.max(predictions, 1)[1].view(target.size()).data == target.data).float().sum()/len(batch)
            # print(torch.max(predictions, 1)[1].view(target.size()).data, target.data, "val")
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_final(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    y_test = []
    pred_test = []
    #deactivating dropout layers
    model.eval()
    model.cuda()
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #retrieve text and no. of words
            text, text_lengths = batch.text
            
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
              text = text.cuda()
              target = target.cuda()
            
            #convert to 1d tensor
            # predictions = model(text, text_lengths).squeeze()
            predictions = model(text, text.size()[0]).squeeze()
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = (torch.max(predictions, 1)[1].view(target.size()).data == target.data).float().sum()/len(batch)
            # print(torch.max(predictions, 1)[1].view(target.size()).data, target.data, "val")
            y_test.append(target.data)
            pred_test.append(torch.max(predictions, 1)[1].view(target.size()).data)
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), y_test, pred_test

size_of_vocab = len(TEXT.vocab)
embedding_dim = 300
hidden_dim = 64
num_output_nodes = 13
num_layers = 4
bidirection = False
dropout = 0.2
print(size_of_vocab)
model = classifier(size_of_vocab, embedding_dim, hidden_dim, num_output_nodes, num_layers, bidirection, dropout)
optimizer = optim.Adam(model.parameters())
criterion = F.cross_entropy
best_valid_acc = 0
N_EPOCHS = 100
best_valid_loss = float('inf')

train_losses = []
val_losses = []
train_accs = []
val_accs = []
for epoch in range(N_EPOCHS):
     
    #train the model
    # print()
    train_loss, train_acc = train(model, train_iter, optimizer, criterion, epoch)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    train_accs.append(train_acc)
    val_accs.append(valid_acc)

    #save the best model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'novel_model_best2.pt')
        torch.save(model, 'novel_model_best2')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%', epoch)

import matplotlib.pyplot as plt

epoch_list = []
for i in range(15):
  epoch_list.append(i + 1)
# plt.plot(epoch_list, train_losses[:10])
plt.plot(epoch_list, train_accs[:15])
plt.plot(epoch_list, val_accs[:15])
# plt.plot(epoch_list, val_losses[:10], '--r')
# plt.legend(["train loss", "validation loss"])
plt.legend(["train Accuracy", "validation Accuracy"])
plt.xlabel("epochs")
# plt.savefig("model1.jpeg", dpi = 500, format = "jpeg")
plt.show()

# Commented out IPython magic to ensure Python compatibility.
model = torch.load('novel_model_best')
import numpy as np
# evaluate the model
valid_loss, valid_acc, y_test, pred_test = eval_final(model, valid_iter, criterion)
print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')

valid_loss, valid_acc, y_test, pred_test = eval_final(model, train_iter, criterion)
print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')

def return_labels(tensor_list):
  new_list = np.array(tensor_list)
  labels = []
  for i in range(new_list.shape[0]):
    labels.append(new_list[i].squeeze().tolist())
  final_labels = []
  for i in range(len(labels)):
    for j in range(len(labels[i])):
      final_labels.append(labels[i][j])
  return final_labels

y_test = return_labels(y_test)
pred_test = return_labels(pred_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_test)

import seaborn as sns
# %matplotlib inline
sns.heatmap(cm, annot=True, cbar=True, cmap = 'YlGnBu', fmt = 'd')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
# plt.title('Confusion Matrix')
plt.savefig("cm_lstm_3.jpeg", dpi = 500, format = 'jpeg')
