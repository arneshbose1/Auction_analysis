import warnings
warnings.filterwarnings("ignore")
import torch 
import torch.nn as nn

class Role_Classifier(nn.Module):
    def __init__(self, config, device):
        super(Role_Classifier, self).__init__()
        
        self.config = config
        self.device = device
        
        self.enc_act = config['encoder_activation']
        self.enc_depth = config['encoder_depth']
        self.enc_drop = config['encoder_dropout']
        self.enc_dim = config['encoder_dimension']
        self.dec_act = config['decoder_activation']
        self.dec_depth = config['decoder_depth']
        self.dec_drop = config['decoder_dropout']
        
        self.act_dict = {'tanh': nn.Tanh(), 'ReLu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
             
        self.enc1_layers = [nn.Linear(5, self.enc_dim), nn.Dropout(self.enc_drop)]
        for i in range(self.enc_depth - 1):
            self.enc1_layers += [nn.Linear(self.enc_dim, self.enc_dim), self.act_dict[self.enc_act], nn.Dropout(self.enc_drop)]
        self.enc1_layers = nn.ModuleList(self.enc1_layers)
            
        self.enc2_layers = [nn.Linear(5, self.enc_dim), nn.Dropout(self.enc_drop)]
        for i in range(self.enc_depth - 1):
            self.enc2_layers += [nn.Linear(self.enc_dim, self.enc_dim), self.act_dict[self.enc_act], nn.Dropout(self.enc_drop)]
        self.enc2_layers = nn.ModuleList(self.enc2_layers)
          
        self.dec_layers = []
        for i in range(self.dec_depth):
            self.dec_layers += [nn.Linear(2*self.enc_dim, 2*self.enc_dim), self.act_dict[self.dec_act], nn.Dropout(self.dec_drop)]
        self.dec_layers = nn.ModuleList(self.dec_layers)
        
        self.fc = nn.Linear(2*self.enc_dim, 3)
        self.role = nn.Softmax()
    

    def forward(self, examples):

        src1 = examples['tendency']
        src2 = examples['career']
        
        for layer in self.enc1_layers:
            src1 = layer(src1)
        
        for layer in self.enc2_layers:
            src2 = layer(src2)
            
        src = torch.cat((src1,src2), dim=1)
        
        for layer in self.dec_layers:
            src = layer(src)

        results = self.fc(src)
        
        return results



          