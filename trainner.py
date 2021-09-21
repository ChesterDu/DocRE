from dgl.batch import batch
import torch.nn as nn
import torch
import dgl
import copy
import fitlog
import tqdm
import torch.nn.functional as F
from model import OUTPUT_NUM

def f1_metric(y_pred,y_true):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1-y_true) * (1-y_pred)).sum().to(torch.float32)
    fp = ((1-y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

#     epsilon = 1e-7
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)
#     f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return tp.item(),tn.item(),fp.item(),fn.item()


class Trainner(nn.Module):
    def __init__(self,config,model,optimizer,criterion):
        super(Trainner,self).__init__()
        
        self.log_pth = config.log_pth
        self.checkpoint_pth = config.checkpoint_pth
        
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.time = 0

        self.total_steps = config.total_steps
        self.epoch = config.epoch
        self.step_count = 0
        self.forward_count = 0
        self.num_acumulation = config.num_acumulation
        self.metric_check_freq = config.metric_check_freq
        self.loss_print_freq = config.loss_print_freq

        self.epoch_count = 1

        self.lr = config.lr
        self.device = config.device
        
        if config.debug:
            fitlog.set_log_dir("../debug_logs")
        else:
            fitlog.set_log_dir(self.log_pth)
        fitlog.add_hyper(config)
        
        self.theta = config.theta
    
    
    def forward_step(self,batch_data):
        out = self.model(batch_data)

        return out
    
    def backward_step(self,logits,labels):
        loss = self.criterion(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        loss.backward()

        for p in self.model.embed.parameters():
            print(p.grad)

        return loss

    def train(self,train_loader,dev_loader):
        print(self.model.parameters)
        self.total_steps = (self.epoch * len(train_loader) - 1) // self.num_acumulation + 1
        bar = tqdm.tqdm(total=self.total_steps)
        bar.update(0)

        # epoch_num = (self.total_steps - 1) // len(train_loader) + 1
        # while(self.step_count < self.total_steps):
        while(self.epoch_count <= self.epoch):
            self.model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                multi_labels = batch_data['multi_label'].to(self.device).reshape(-1,logits.shape[-1]).to(torch.float)
                loss = self.criterion(logits,multi_labels) # reduction = none, shape: [bsz * h_t_pair_num]
                label_mask = batch_data['label_mask'].to(self.device).reshape(-1)
                loss = loss.sum(dim=-1) * label_mask
                loss = loss.sum() / label_mask.sum()

                # na_indices = (labels == 0).nonzero().squeeze(-1)
                # ignore_indices = (labels == -1).nonzero().squeeze(-1)
                # num_na = na_indices.shape[0]
                # num_pos = labels.shape[0] - num_na
                
                # loss_weight = torch.FloatTensor([1/num_pos * labels.shape[0]]).to(self.device)
                # loss_weight[na_indices] = 1/num_na
                # loss_weight[ignore_indices] = 0

                
                # loss = torch.mul(loss,loss_weight)
                # loss = torch.sum(loss) / (labels.shape[0] - ignore_indices.shape[0])



                epoch_loss += loss.item()
                loss.backward()
                self.forward_count += 1

                if (self.forward_count % self.num_acumulation) == 0:
                    self.optimizer.step()
                    self.step_count += 1
                    bar.update(1)
                    fitlog.add_loss(loss.item(),name='Loss',step=self.step_count)
                    self.optimizer.zero_grad()
                    if self.step_count >= self.total_steps:
                        break


                    if (self.step_count % self.loss_print_freq) == 0:
                        print("Epoch:{}/{} || Step:{}/{} || Loss:{}".format(self.epoch_count,self.epoch,self.step_count,self.total_steps,epoch_loss/self.forward_count))
                    
                    # if (self.step_count % self.metric_check_freq) == 0:
            print('Evaluation Start......')
            p,r,f1 = self.evaluate_multi(dev_loader)
            print("Eval Results Epoch: {} || Step:{}/{} || Precision: {} || Recall: {} || F1: {}".format(self.epoch_count,self.step_count,self.total_steps,p,r,f1))
            fitlog.add_metric({"dev":{"Precision":p}}, step=self.step_count,epoch=self.epoch_count)
            fitlog.add_metric({"dev":{"Recall":r}}, step=self.step_count,epoch=self.epoch_count)
            fitlog.add_metric({"dev":{"F1":f1}}, step=self.step_count,epoch=self.epoch_count)
            # fitlog.add_metric({"dev":{"Loss":test_loss}},step=self.step_count,epoch=self.epoch_count)
            self.epoch_count += 1
    

    def evaluate_multi(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            total_tp = 0
            total_tn = 0
            total_fp = 0
            total_fn = 0
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['multi_label'].to(self.device).reshape(-1,logits.shape[-1]).to(torch.float)
                predections_re = (torch.sigmoid(logits) > self.theta).to(torch.float)
                
                indices = (labels != -1).nonzero().squeeze(-1)
                labels = labels[indices]
                predections_re = predections_re[indices]
                
                tp,tn,fp,fn = f1_metric(predections_re,labels)
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

             
            epsilon = 1e-7
            p = total_tp / (total_tp + total_fp + epsilon)
            r = total_tp / (total_tp + total_fn + epsilon)
            f1 = 2* (p*r) / (p + r + epsilon)
        return p,r,f1

        
    def evaluate(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            total_na_pred = None
            total_na_label = None
            total_nonNa_pred = None
            total_nonNa_label = None
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['label'].to(self.device).reshape(-1).to(torch.float)
                # loss = self.criterion(logits,labels)
                # test_loss.append(loss.item())

                prediction = torch.argmax(logits,dim=1)
                indices = (labels != -1).nonzero().squeeze(-1)
                labels = labels[indices]
                prediction = prediction[indices]

                na_indices = (labels == 0).nonzero().squeeze(-1)
                na_pred = F.one_hot((prediction[na_indices] == 0).to(torch.int64),2)
                na_label = F.one_hot((labels[na_indices] == 0).to(torch.int64),2)

                nonNa_indices = (labels != 0).nonzero().squeeze(-1)
                nonNa_pred = F.one_hot(prediction[nonNa_indices],OUTPUT_NUM)
                nonNa_label = F.one_hot(labels[nonNa_indices],OUTPUT_NUM)

                total_na_pred = torch.cat([total_na_pred,na_pred],dim=0) if total_na_pred != None else na_pred
                total_na_label = torch.cat([total_na_label,na_label],dim=0) if total_na_label != None else na_label
                total_nonNa_pred = torch.cat([total_nonNa_pred,nonNa_pred],dim=0) if total_nonNa_pred != None else nonNa_pred
                total_nonNa_label = torch.cat([total_nonNa_label,nonNa_label],dim=0) if total_nonNa_label != None else nonNa_label

            def f1_metric(y_pred,y_true):
                tp = (y_true * y_pred).sum().to(torch.float32)
                tn = ((1-y_true) * (1-y_pred)).sum().to(torch.float32)
                fp = ((1-y_true) * y_pred).sum().to(torch.float32)
                fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2* (precision*recall) / (precision + recall + epsilon)

                return precision.item(),recall.item(),f1.item()
            nonNa_p,nonNa_r,nonNa_f1 = f1_metric(total_nonNa_pred,total_nonNa_label)
            na_p,na_r,na_f1 = f1_metric(total_na_pred,total_na_label)

        return {"nonNa_p":nonNa_p,"nonNa_r":nonNa_r,"nonNa_f1":nonNa_f1,"na_p":na_p,"na_r":na_r,"na_f1":na_f1}

