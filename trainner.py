from dgl.batch import batch
import torch.nn as nn
import torch
import dgl
import copy
import fitlog
import tqdm


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

        for p in self.model.embed.parameters():
            print(p.grad)

        return loss

    def train(self,train_loader,dev_loader):
        self.model.init_params()
        bar = tqdm.tqdm(total=self.total_steps)
        bar.update(0)
        while(self.step_count < self.total_steps):
            self.model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['label'].to(self.device).reshape(-1)
                loss = self.criterion(logits,labels)

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
                        print("Step:{}/{} || Loss:{}".format(self.step_count,self.total_steps,epoch_loss/self.forward_count))
                    
                    # if (self.step_count % self.metric_check_freq) == 0:
            print('Evaluation Start......')
            test_total_acc, test_na_acc, test_non_na_acc = self.evaluate(dev_loader)
            print("Eval Results Epoch: {} || Step:{}/{} || Acc: {} || NA Acc: {} || Non NA Acc: {}".format(self.epoch_count,self.step_count,self.total_steps,test_total_acc, test_na_acc, test_non_na_acc))
            fitlog.add_metric({"dev":{"Acc":test_total_acc}}, step=self.step_count,epoch=self.epoch_count)
            fitlog.add_metric({"dev":{"NA Acc":test_na_acc}}, step=self.step_count,epoch=self.epoch_count)
            fitlog.add_metric({"dev":{"Non NA Acc":test_non_na_acc}}, step=self.step_count,epoch=self.epoch_count)
            # fitlog.add_metric({"dev":{"Loss":test_loss}},step=self.step_count,epoch=self.epoch_count)
            self.epoch_count += 1
    

    def evaluate(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            test_loss = []
            correct_pred = 0
            total_pred = 0
            total_na_pred = 0
            na_correct_pred = 0
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['label'].to(self.device).reshape(-1)
                # loss = self.criterion(logits,labels)
                # test_loss.append(loss.item())

                prediction = torch.argmax(logits,dim=1)
                indices = (labels != -1).nonzero().squeeze(-1)
                labels = labels[indices]
                prediction = prediction[indices]

                total_pred += prediction.shape[0]
                correct_pred += torch.sum(prediction == labels).item()

                na_indices = (labels == 0).nonzero().squeeze(-1)
                na_pred = prediction[na_indices]
                
                na_correct_pred += torch.sum(na_pred == 0).item()
                total_na_pred += na_pred.shape[0]


            # test_loss = sum(test_loss) / len(test_loss)
            test_total_acc = correct_pred / total_pred
            test_na_acc = na_correct_pred / total_na_pred
            test_non_na_acc = (correct_pred - na_correct_pred) / (total_pred - total_na_pred)

        return test_total_acc, test_na_acc, test_non_na_acc

