from dgl.batch import batch
import torch.nn as nn
import torch
import dgl
import copy
import fitlog


class trainner(nn.Module):
    def __init__(self,args,model,optimizer,criterion):
        self.log_pth = args.log_pth
        self.checkpoint_pth = args.checkpoint_pth
        
        self.model = copy.deepcopy(model).to(args.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.time = 0

        self.total_steps = args.total_steps
        self.step_count = 0
        self.forward_count = 0
        self.num_acumulation = args.num_acumulation
        self.metric_check_freq = args.metric_check_feq
        self.loss_print_freq = args.loss_print_freq

        self.epoch_count = 1

        self.lr = args.lr
        self.device = args.device

        fitlog.set_log_dir(self.log_pth)
        fitlog.add_hyper(args)
    
    
    def forward_step(self,batch_data):
        batched_x = self.model.embed(batch_data['token_id'].to(self.device))

        batched_node_features = None
        batched_edge_features = None
        node_offset = 0
        for i,sample in enumerate(batch_data['sample']):
            node_features, edge_features = self.model.init_node_edge_features(batched_x[i],sample)
            batched_node_features = torch.cat([batched_node_features,node_features],dim=0) if batched_node_features != None else node_features
            batched_edge_features = torch.cat([batched_edge_features,edge_features],dim=0) if batched_edge_features != None else edge_features
            batch_data['headEnt'][i] += node_offset
            batch_data['tailEnt'][i] += node_offset

            node_offset += len(sample['graphData']['nodes'])
        
        batched_g = dgl.batch(batch_data['graph']).to(self.device)
        out = self.model(batched_g,batched_node_features,batched_edge_features,batch_data['headEnt'].to(self.device),batch_data['tailEnt'].to(self.device))

        return out
    
    def backward_step(self,logits,labels):
        loss = self.criterion(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        loss.backward()

        return loss

    def train(self,train_loader,dev_loader,test_loader):
        while(self.step_count < self.total_steps):
            for batch_data in train_loader:
                logits = self.forward_step(batch_data)
                loss = self.backward_step(logits,batch_data['labels'].to(self.device))
                self.forward_count += 1

                if (self.forward_count % self.num_acumulation) == 0:
                    self.optimizer.step()
                    self.step_count += 1
                    fitlog.add_loss(loss.item(),name='Loss',step=self.step_count)
                    self.optimizer.zero_grad()
                    if self.step_count >= self.total_steps:
                        break


                    if (self.step_count % self.loss_print_freq) == 0:
                        print("Step:{}/{} Loss:{}".format(self.step_count,self.total_steps,loss.item()))
                    
                    if (self.step_count % self.metric_check_freq) == 0:
                        test_acc,test_loss = self.evaluate(dev_loader)
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
                labels = batch_data['labels'].to(self.device).reshape(-1)
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

