from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
import torch

class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    
    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r
        
        x = self.interaction_emb(x)
        h, _ = self.lstm_layer(x)
        
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(self, train_loader, num_epochs, opt, verbose = False):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                num_epochs: the number of epochs
                opt: the optimization to train this model
        '''
        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data
                
                self.train()

                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())
            
            if verbose:
                print(f'epoch {i} : {sum(loss_mean)/len(loss_mean)}')
    
    def test_model(self, test_loader):
        res = torch.Tensor()
        
        for data in test_loader:
            q, r, m = data
            
            m = m.long().sum(1)-1
            
            self.eval()
            
            y = self(q.long(), r.long()).reshape(-1, self.num_q)
            y = y[m,:]
            
            res = torch.cat((res,y),0)
        
        return res