import os
import tqdm
from time import time

import torch
import torch.nn as nn

import torchvision

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils import Utils
from metrics import Metrics

class DiffTrainer(Utils):
    # Init params for train and test
    def __init__(self):
        self.root="/home/kaist/inje/sr/data"
        self.img_num=50000 # CIFAR 10 Train Data num
        self.train_batch_size=128
        self.eval_batch_size=256
        self.workers=4
        self.report_img_per = 100
        self.report_img_num = 10
        self.report_img_size = 32
    
    def setup_and_test(self, virtual_device, ngpus_per_node, settings):
        self.setup_exec_env(virtual_device, ngpus_per_node, settings)
        self._setup_eval_env(virtual_device, ngpus_per_node)

        # do test()
        self._test_network()
        return

    def setup_and_train(self, virtual_device, ngpus_per_node, settings, resume):
        self.setup_exev_env(virtual_device, ngpus_per_node, settings)
        self._setup_eval_env(virtual_device, ngpus_per_node)
        self._setup_train_env()

        # do train
        self._train_network(resume)
        return
    
    # Setup evalutaion environment; used for validation and test 
    def _setup_eval_env(self, virtual_device, ngpus_per_node):
        # define dataloader
        eval_transform = transforms.ToTensor()
        eval_dataset = torchvision.datasdts.CIFAR10(root=self.root, train=True, transform=eval_transform)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False) if self.mgpu and self.is_divisible(eval_dataset, ngpus_per_node) else None
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, num_workers=self.workers, sampler=eval_sampler, pin_memory=True)
        
        # define scores
        self.scores = self.define_model(
            net=Metrics(self.device), \
            ngpus_per_node=ngpus_per_node, \
            virtual_device=virtual_device, \
            device=self.device, \
            master=self.master, \
            mgpu=self.mgpu, \
            addr=self.addr, \
            port=self.port \
        )

        # setup scores
        self.scores.setup(self.eval_dataloader)
        return
    
    # Setup train environment
    def _setup_train_env(self):
        # defince dataloader
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
        train_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=train_transform)
        train_sampler = DistributedSampler(train_dataset, shuffle=False, drop_last=True) if self.mgpu else None
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.workers, sampler=train_sampler, pin_memory=True)

        # register train variables 
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.iter_per_epoch = len(train_dataloader)
        self.epochs = (self.iters + iter_per_epoch - 1) // iter_per_epoch
        self.scheduler = LambdaLR(self.optimizer, lr_lambda= lambda step: min(step, 5000)/5000)
        self.sample_x_T = torch.randn(self.report_img_num, 3, self.report_img_size, self.report_img_size).to(self.device)

        return
    
    def _test_network(self):
        # load test env
        self.load_test_env(self.ema_net, self.check_pnt_path, mgpu=self.mgpu, user_set_devices=self.user_set_devices):
        # evaluate scores
        (is_mean, is_std), fid, sample = self._eval(fix_sample=False)

        self.writer.add_scalars('Test/IS', {'mean': is_mean, 'std':is_std}, 0)
        self.writer.add_scalar('Test/FID', fid, 0)
        self.writer.add_images('Test/Images', sample, 0)
        print(f'------------------------------------------------------------------------')
        print(f'Test: [IS] ({is_mean:.3f}, {is_std:.3f}), [FID] {fid} | Time: {time:.3f}')

        return
    
    def _train_network(self, resume):
        # load train env
        epoch, best_score = self.load_train_env(self.net, self.ema_net, self.optimizer, self.scheduler, resume, \
                                                path=self.last_pnt_path, mgpu=self.mgpu, user_set_devices=self.user_set_devices)

        while epoch < self.epochs:

            start = time()

            t_loss =  self._train(epoch)
            is_mean, is_std, fid, v_loss, sample = self._valid(epoch)

            if self.master:
                self.store_train_env(self, epoch, self.net, self.ema_net, self.optimizer, self.scheduler, fid, best_score, \
                                    old_pnt_path, last_pnt_path, check_pnt_path, self.mgpu)

            end = time()

            time = end-start

            if self.master:
                self.writer.add_scalar('Train/Loss', loss_t, epoch)
                self.writer.add_scalar('Valid/Loss', loss_v, epoch)
                self.writer.add_scalars('Valid/IS', {'mean': is_mean, 'std':is_std}, epoch)
                self.writer.add_scalar('Valid/FID', fid, epoch)
                if epoch%self.report_img_per == 0:
                    self.writer.add_images('Valid/Images', sample, epoch)
                print(f'Epoch: {epoch+1}/{self.epochs} | Train: [Loss] {t_loss:.3f} | Valid: [Loss] {v_loss:.3f}, [IS] ({is_mean:.3f}, {is_std:.3f}), [FID] {fid:.3f} | Time: {time:.3f}')
            
            epoch += 1
        return

    # Train code
    def _train(self, epoch):
        # start training
        self.net.train()
        self.ema_net.train()

        t_loss_tot = 0.0
        for img in tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]'):
            
            self.optimizer.zero_grad()

            img = self.norm(img).to(self.device)
            loss = self.net(img)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1) # prevent gradient exploding
            self.optimizer.step()
            self.scheduler.step() 

            t_loss_tot += loss.item()

            # update ema_net
            self.exec_ema(self.net, self.ema_net)        
        
        # collect loss
        if self.mgpu:
            dist.all_reduce(t_loss_tot, op=dist.ReduceOp.SUM)
            N = dist.get_world_size()
            t_loss_tot /= N

        # average loss
        t_loss_avg = t_loss_tot/len(self.train_dataloader)

        train_time = end-start
        return t_loss_avg
    
    # Validatiion code
    def _valid(self, epoch):
        # start evaluation
        self.net.eval()
        self.ema_net.eval()
        
        v_loss_tot = 0.0

        with torch.no_grad():
            for img in tqdm(self.eval_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} [Valid]'):

                img = self.norm(img).to(self.device)
                loss = self.ema_net(img)
                v_loss_tot += v_loss.item()

            # collect loss
            if self.mgpu:
                N = dist.get_world_size()
                dist.all_reduce(v_loss_tot, op=dist.ReduceOp.SUM)
                v_loss_tot /= N

            # average loss
            v_loss_avg = v_loss_tot/len(self.eval_dataloader)
            
            # evaluate scores
            (is_mean, is_std), fid, sample = self._eval(fix_sample=True)
        
        return is_mean, is_std, fid, v_loss_avg, sample

    # Score evaluation code; used for validation and test
    def _eval(self, fix_sample):
        self.ema_net.eval()
        with torch.no_grad():
            # calculate scores
            iters = (self.img_num+self.eval_batch_size-1)//self.eval_batch_size
            c, h, w = 3, self.report_img_size, self.report_img_size
            for i in range(iters):
                b = min(self.eval_batch_size, self.img_num-self.eval_batch_size*i)
                x_T = torch.randn(b, c, h, w).to(self.device)
                x_0 = self.ema_net.sample(x_T)
                self.scores.update(x_0)
            # get sample images
            sample = x_0 if not fix_sample else self.ema_net.sample(self.sample_x_T)
        
        if self.mgpu:
            N = dist.get_world_size()
            # collect activations
            is_acts, fid_acts = self.scores.get_activations()
            is_acts_gather = [torch.ones_like(is_acts) for _ in range(N)]
            fid_acts_gather = [torch.ones_like(fid_acts) for _ in range(N)]
            
            dist.all_gather(is_acts_gather, id_acts)
            dist.all_gather(fid_acts_gather, fid_acts)

            is_acts = torch.cat(is_acts_gather)
            fid_acts = torch.cat(fid_acts_gather)
            self.scores.set_activations(is_acts, fid_acts)
        
        return self.scores(), sample
    
    
    
    
    
