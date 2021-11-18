from torch import Tensor
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn import Parameter
torch.cuda.empty_cache()

#RBF
def gaussian(alpha): return -0.5*alpha 

class Model(nn.Module):
    def __init__(self, num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, drop_rate, variance_x, bs, 
                 embedding_words, word_emb_size,activation,distance="gaussian"):
      
        super(Model, self).__init__()
        self.num_input, self.num_coordinate, self.num_topic, self.variance_x, self.bs \
            = num_input, num_coordinate, num_topic, variance_x, bs

        self.embedding_words = embedding_words
        self.emb_size = word_emb_size
        self.activation = activation
        self.embedding_words = embedding_words
        self.emb_size = word_emb_size
 
        # encoder
        self.en1_fc     = nn.Linear(num_input, en1_units_x) 
        self.en2_fc     = nn.Linear(en1_units_x, en2_units_x)
        self.mu1_fc     = nn.Linear(2, 100) 
        self.mu2_fc     = nn.Linear(100, 100)
        self.mu_fc      = nn.Linear(100, 300)
        self.mu_z = 0

        self.en2_drop   = nn.Dropout(drop_rate)
        self.mean_fc    = nn.Linear(en2_units_x, num_coordinate) 
        self.logvar_fc  = nn.Linear(en2_units_x, num_coordinate) 

        self.mean_bn    = nn.BatchNorm1d(num_coordinate)                    
        self.logvar_bn  = nn.BatchNorm1d(num_coordinate)
        self.decoder_x_bn = nn.BatchNorm1d(num_coordinate)    
        self.decoder_phi_bn = nn.BatchNorm1d(num_coordinate) 
        self.decoder_bn = nn.BatchNorm1d(self.num_topic)     
        self.mu_z_bn = nn.BatchNorm1d(self.emb_size)                          

        # RBF
        self.in_features = self.num_coordinate
        self.out_features = self.num_topic
        self.topics = nn.Parameter(torch.Tensor(self.out_features, self.in_features)) # K x 2
        self.beta_bias = nn.Parameter(torch.Tensor(self.out_features,self.num_input))
              
        if distance=="gaussian": self.basis_func = gaussian
        if distance=="inverse_quadratic": self.basis_func = inverse_quadratic
        self.init_parameters()
               
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_coordinate).fill_(0)
        prior_var    = torch.Tensor(1, num_coordinate).fill_(variance_x)
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        self.prior_var  = nn.Parameter(prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(prior_var.log(), requires_grad=False)

    def init_parameters(self):
        nn.init.normal_(self.topics, 0, 0.1)
        nn.init.normal_(self.beta_bias, 0, 0.01)

    def get_activation(self, activation,layer):
      activation = activation.lower()
      if activation == 'relu':
          layer =  F.relu(layer)
      elif activation == 'softplus':
          layer =  F.softplus(layer)
      elif activation == 'sigmoid':
          layer =  F.sigmoid(layer)
      elif activation == 'leaky_relu':
          layer = F.leaky_relu(layer)
      else:
          layer = F.softplus(layer)
      return layer 
      
    def encode(self, input_,normalized_input_):
        N, *_ = input_.size()
        
        en1 = self.get_activation(self.activation,self.en1_fc(input_))                         
        en2 = self.get_activation(self.activation,self.en2_fc(en1))                           
        
        # en1 = F.softplus(self.en1_fc(input_))                      
        # en2 = F.softplus(self.en2_fc(en1))   
        
        en2 = self.en2_drop(en2)

        posterior_mean   = self.mean_bn(self.mean_fc(en2))        
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))      
        posterior_var    = posterior_logvar.exp()
        
        return en2, posterior_mean, posterior_logvar, posterior_var

    def take_sample(self, input_, posterior_mean, posterior_var, prior_var):
        eps = input_.data.new().resize_as_(posterior_mean.data).normal_(std=1.0) # noise(epsilon)
        z = posterior_mean + posterior_var.sqrt() * eps     # reparameterization trick
        return z

    def get_beta(self): 
        return F.softmax(self.decoder_bn(torch.mm(self.mu_z,self.embedding_words.T).T).T + self.beta_bias,dim=-1)

    def decode(self, z):

      ## Theta - P(z|x,phi) ## NXT
      N, *_ = z.size()
      size = (N, self.out_features, self.in_features) # N,T,2

      ## apply batch normalization across X  (Document coordintes)
      zx = self.decoder_x_bn(z) # Nx2
      x = zx.view(N, 1, self.num_coordinate).expand(size) # Nx1x2

      ## apply batch normalization across phi (Topic coordintes)
      zc = self.decoder_phi_bn(self.topics)
      c = zc.view(1, self.num_topic, self.num_coordinate).expand(size)

      d = (x-c).pow(2).sum(-1)
      distances = self.basis_func(d)
      theta = torch.exp(distances - torch.logsumexp(distances, dim=-1, keepdim=True)) # N x T

      ## Topic Coordinates mapped to topic embeddings
      mu1 = F.softplus((self.mu1_fc(zc)))                          
      mu2 = F.softplus(self.en2_drop(self.mu2_fc(mu1)))
      self.mu_z = self.mu_fc(mu2)

      ## Beta P(w|z) ## (TxV)
      beta = self.get_beta()
      recon_v = torch.mm(theta,beta)

      return recon_v, zx, theta,zc

    
    def forward(self, input_,normalized_input_, compute_loss=False):  
        en2, posterior_mean, posterior_logvar, posterior_var = self.encode(input_,normalized_input_)
        z = self.take_sample(input_, posterior_mean, posterior_var, self.variance_x)
        recon_v, zx, theta,zc= self.decode(z)
        
        if compute_loss:
            return recon_v, zx,self.loss(input_, recon_v, theta, posterior_mean, posterior_logvar, posterior_var, zx)
        else: return z, recon_v, zx, zc, theta
 
    def KLD(self, posterior_mean,posterior_logvar,posterior_var):
        N = posterior_mean.shape[0]
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)

        var_division    = posterior_var  / prior_var 
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(-1) - self.num_coordinate) 
        return KLD
 
    def loss(self, input_, recon_v, theta, posterior_mean, posterior_logvar, posterior_var, zx,avg=True):
        N = posterior_mean.shape[0]
        smoothen = 1e-6 # 'error/term-smoothening constant'

        NL = - (input_ * (recon_v+smoothen).log()).sum(-1)
        NL= NL.mean(0)
     
        KLD = self.KLD(posterior_mean,posterior_logvar,posterior_var).mean(0)

        loss = NL + KLD
        return loss,NL,KLD
