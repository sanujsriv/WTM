import torch.optim as optim 
import torch
import numpy as np 

def train_for_large(model,count_vec,train_label,num_input,num_topic,learning_rate,beta1,
                   beta2,all_indices,epochs,device):
  
  kld_arr,recon_arr = [],[]
  model.to(device)

  optimizer = optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
  for epoch in range(epochs):

      loss_u_epoch = 0.0 ## NL loss
      loss_KLD = 0.0  ## KL loss
      loss_epoch = 0.0 ## Loss per batch #
      
      model.train()
      zx_l = []
      label_l = []
      for batch_ndx in all_indices:

        input_w = torch.from_numpy(count_vec[batch_ndx].toarray()).float().to(device)
        normalized_inputw = input_w/(input_w.sum(1).unsqueeze(1))
        labels = train_label[batch_ndx]
        label_l.extend(labels)
        recon_v, zx,(loss, loss_u, xkl_loss) = model(input_w,normalized_inputw,compute_loss=True)
        zx_l.extend(zx.data.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()             # backpror.step()        
        loss_epoch += loss.item()
        loss_u_epoch += loss_u.item()
        loss_KLD += xkl_loss.item()
        current_model = model 
      kld_arr.append(loss_KLD)
      recon_arr.append(loss_u_epoch)

      if epoch % 10 == 0:
          # print('Epoch -> {}'.format(epoch))
          print('Epoch -> {} , loss -> {}'.format(epoch,loss_epoch))
          print('recon_loss==> {} || KLD==> {}'.format(loss_u_epoch, loss_KLD))
          # plot_fig(np.array(zx_l),label_l,model.decoder_phi_bn(model.centres).data.cpu().numpy(),10.0,'No')
  return current_model

def test_for_large(model,all_indices,count_vec,num_topic,train_label,id_vocab,device):
  model.eval()
  x_list = []
  labels_list = []
  doc_ids = []
  zx_phi_list=[]
  with torch.no_grad():
      for batch_ndx in all_indices:
          input_w = torch.from_numpy(count_vec[batch_ndx].toarray()).float().to(device)
          normalized_inputw = input_w/(input_w.sum(1).unsqueeze(1))
          labels = train_label[batch_ndx] 
          labels_list.extend(labels)
          
          z, recon_v, zx,zphi,zx_phi = model(input_w,normalized_inputw,compute_loss=False)
          zx = zx.data.detach().cpu().numpy()
          zphi = zphi.data.detach().cpu().numpy()
          zx_phi = zx_phi.view(-1, num_topic).data.detach().cpu().numpy()
          zx_phi_list.extend(zx_phi)
          x_list.extend(zx)
          doc_ids.extend(batch_ndx)
          
      x_list = np.array(x_list)
      beta = model.get_beta().data.cpu().numpy()

      # zphi = model.decoder_phi_bn(model.centres).data.cpu().numpy()
      # # zphi = model.centres.data.cpu().numpy()
      
  return x_list,labels_list,zphi,doc_ids,beta

def train(model,tensor_train_w,train_label,num_input,num_topic,learning_rate,beta1,
                   beta2,all_indices,epochs,device):
  
  kld_arr,recon_arr = [],[]
  model.to(device)

  optimizer = optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
  for epoch in range(epochs):

      loss_u_epoch = 0.0 ## NL loss
      loss_KLD = 0.0  ## KL loss
      loss_epoch = 0.0 ## Loss per batch #
      
      model.train()
      zx_l = []
      label_l = []
      for batch_ndx in all_indices:
      # batch_ndx = all_indices[0]
        input_w = tensor_train_w[batch_ndx].to(device)
        normalized_inputw = input_w/(input_w.sum(1).unsqueeze(1))
        labels = train_label[batch_ndx]
        label_l.extend(labels)
        recon_v, zx,(loss, loss_u, xkl_loss) = model(input_w,normalized_inputw,compute_loss=True)
        zx_l.extend(zx.data.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()             # backpror.step()        
        loss_epoch += loss.item()
        loss_u_epoch += loss_u.item()
        loss_KLD += xkl_loss.item()
        current_model = model 
      kld_arr.append(loss_KLD)
      recon_arr.append(loss_u_epoch)

      if epoch % 10 == 0:
          #  print('Epoch -> {}'.format(epoch))
          print('Epoch -> {} , loss -> {}'.format(epoch,loss_epoch))
          print('recon_loss==> {} || KLD==> {}'.format(loss_u_epoch, loss_KLD))
          # plot_fig(np.array(zx_l),label_l,model.decoder_phi_bn(model.centres).data.cpu().numpy(),10.0,'No')
  return current_model

def test(model,all_indices,tensor_train_w,num_topic,train_label,id_vocab,device):
  model.eval()
  x_list = []
  labels_list = []
  doc_ids = []
  zx_phi_list=[]
  with torch.no_grad():
      for batch_ndx in all_indices:
          input_w = tensor_train_w[batch_ndx].to(device)
          normalized_inputw = input_w/(input_w.sum(1).unsqueeze(1))
          labels = train_label[batch_ndx] 
          labels_list.extend(labels)
          
          z, recon_v, zx,zphi, zx_phi = model(input_w,normalized_inputw,compute_loss=False)
          zx = zx.data.detach().cpu().numpy()
          zphi = zphi.data.detach().cpu().numpy()
          zx_phi = zx_phi.view(-1, num_topic).data.detach().cpu().numpy()
          zx_phi_list.extend(zx_phi)
          x_list.extend(zx)
          doc_ids.extend(batch_ndx)
          
      x_list = np.array(x_list)
      beta = model.get_beta().data.cpu().numpy()

  return x_list,labels_list,zphi,doc_ids,beta