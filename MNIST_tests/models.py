import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import linear

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class simpleExpandedDNN(nn.Module):
  def __init__(self, batch_size, batch_proc_size, input_dim, output_dim, device='cpu'):
    super(simpleExpandedDNN, self).__init__()
    #self.lrelu = nn.LeakyReLU()
    self.latent_dim = 512 # Note: overwritten by BO if used
    self.relu = nn.ReLU()
    self.n_hidden_layers = 1 # number of units/layer (same for all) is set in bo parameters
    self.batch_proc_size = batch_proc_size
    self.batch_size = batch_size
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.linears = nn.ModuleList([ linear.Linear(input_dim, self.latent_dim, bias=True, batch_size=batch_proc_size)])
    if self.n_hidden_layers > 0:
      for k in range(self.n_hidden_layers):
        self.linears.append( linear.Linear(self.latent_dim, self.latent_dim,bias=True,batch_size=batch_proc_size) )
    self.final_fc = linear.Linear(self.linears[-1].out_features, output_dim,bias=True, batch_size=batch_proc_size)

  def forward(self, x):
    x = torch.unsqueeze(x.view(-1, self.input_dim),1)

    for k_linear in self.linears:
      x = self.relu(k_linear(x))
    x = self.final_fc(x)
    
    return nn.functional.log_softmax(x.view(-1,self.output_dim),dim=1)


class LogisticRegression(nn.Module):
    def __init__(self, batch_size, batch_proc_size, input_dim, output_dim, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.batch_proc_size = batch_proc_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
                # Flatten(),
                linear.Linear(input_dim, output_dim, bias=True, batch_size=batch_proc_size)
            ).to(device)

    def forward(self, x):
        x = torch.unsqueeze(x.view(-1, self.input_dim),1)
        return torch.squeeze(self.model(x))


class TwoLayer(nn.Module):
    def __init__(self, batch_size, batch_proc_size, input_dim, output_dim, device='cpu'):
        super(TwoLayer, self).__init__()
        self.batch_proc_size = batch_proc_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            # Flatten(),
            linear.Linear(input_dim, 100, bias=True, batch_size=batch_proc_size),
            linear.Linear(100, output_dim, bias=True, batch_size=batch_proc_size),
        ).to(device)


    def forward(self, x):
        x = torch.unsqueeze(x.view(-1, self.input_dim),1)
        return nn.functional.log_softmax(self.model(x).view(-1,self.output_dim),dim=1)



def get_model(architecture, batch_size, batch_proc_size, input_dim, output_dim, device):

    if(architecture=='LR'):
        classifier = LogisticRegression(batch_size=batch_size, batch_proc_size=batch_proc_size,
                                    input_dim=input_dim, output_dim = output_dim, device = device) 
        loss_function = nn.CrossEntropyLoss()

    elif(architecture=='TLNN'):
        classifier = TwoLayer(batch_size=batch_size, batch_proc_size=batch_proc_size,
                    input_dim=input_dim, output_dim = output_dim, device=device)
        loss_function = nn.NLLLoss()
    elif(architecture=='SEDNN'):
        classifier = simpleExpandedDNN(batch_size=batch_size, batch_proc_size=batch_proc_size,
                    input_dim=input_dim, output_dim = output_dim, device=device)
        loss_function = nn.NLLLoss()
    return classifier, loss_function
