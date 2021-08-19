from dgl.batch import batch
import torch.nn as nn
import torch
import dgl
import copy
import fitlog
import tqdm


class Trainner():
    def __init__(self,config,model,optimizer,criterion):
        self.log_pth = config.log_pth
        self.checkpoint_pth = config.checkpoint_pth
        
        self.model = copy.deepcopy(model).to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.time = 0

        self.total_steps = config.total_steps
        self.step_count = 0
        self.forward_count = 0
        self.num_acumulation = config.num_acumulation
        self.metric_check_freq = config.metric_check_freq
        self.loss_print_freq = config.loss_print_freq

        self.epoch_count = 1

        self.lr = config.lr
        self.device = config.device

        fitlog.set_log_dir(self.log_pth)
        fitlog.add_hyper(config)
    
    
    def forward_step(self,batch_data):
        out = self.model(batch_data)

        return out
    
    def backward_step(self,logits,labels):
        loss = self.criterion(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        loss.backward()

        return loss

    def train(self,train_loader,dev_loader):
        self.model.init_params()
        bar = tqdm.tqdm(total=self.total_steps)
        bar.update(0)
        while(self.step_count < self.total_steps):
            for batch_data in train_loader:
                logits = self.forward_step(batch_data)
                loss = self.backward_step(logits,batch_data['label'].to(self.device))
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
                        print("Step:{}/{} Loss:{}".format(self.step_count,self.total_steps,loss.item()))
                    
                    if (self.step_count % self.metric_check_freq) == 0:
                        test_acc,test_loss = self.evaluate(dev_loader)
                        print("Eval Results Step:{}/{} Loss:{} Acc: {}".format(self.step_count,self.total_steps,test_loss,test_acc))
                        fitlog.add_metric({"dev":{"Acc":test_acc}}, step=self.step_count)
                        fitlog.add_metric({"dev":{"Loss":test_loss}}, step=self.step_count)
    

    def evaluate(self,test_loader):
        with torch.no_grad():
            test_loss = []
            correct_pred = 0
            total_pred = 0
            for batch_data in test_loader:
                logits = self.forward_step(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['label'].to(self.device).reshape(-1)
                loss = self.criterion(logits,labels)
                test_loss.append(loss.item())

                prediction = torch.argmax(logits,dim=1)
                indices = (labels != -1).nonzero()
                labels = labels[indices]
                prediction = prediction[indices]

                total_pred += prediction.shape[0]
                correct_pred += torch.sum(prediction == labels).item()

            test_loss = sum(test_loss) / len(test_loss)
            test_acc = correct_pred / total_pred

        return test_acc, test_loss

