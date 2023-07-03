import os
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion import GaussianDiffusionTrainer

class Utils():
    # Setup ececution environment
    def setup_exec_env(self, virtual_device, ngpus_per_node, settings):

        settings['master'] = virtual_device == 0
        settings['writer'] = SummaryWriter(settings['log_path']) if settings['master'] else None
        settings['device'] = settings['user_set_devices'][virtual_device] if settings['user_set_devices'] != None else virtual_device
        
        net = settings['model']() if settings['args'] == None else settings['model'](*settings['args'])
        net = GaussianDiffusion(net)

        settings['net'] = self.define_model(
            net=net, \
            ngpus_per_node=ngpus_per_node, \
            virtual_device=virtual_device, \
            device=settings['device'], \
            master=settings['master'], \
            mgpu=settings['mgpu'], \
            addr=settings['addr'], \
            port=settings['port'] \
        )

        settings['ema_net'] = copy.deepcopy(net)
        
        for k, v in settings.items():
            setattr(self, k, v)

        return
    
    # Exponential Moving Average
    def exec_ema(self, train_net, sample_net, decay=0.9999):
        train_net_modules = train_net.state_dict()
        sample_net_modules = sample_net.state_dict()

        for name, train_net_module in train_net.items():
            sample_net_module = sample_net_modules[name]
            old = sample_net_module.data
            new = train_net_module.data
            sample_net_modules[key].data.copy_(old*decay + new*(1-decay))

        return

    # Load train environment
    def load_train_env(self, net, ema_net, optimizer, scheduler, resume, path, mgpu=False, user_set_devices=None):
        if resume:
            states = self.get_states(path, mgpu, user_set_devices)
            self._load_model(net, states['net'], mgpu)
            self._load_model(ema_net, states['ema_net'], mgpu)
            optimizer.load_state_dict(states['optimizer'])
            scheduler.load_state_dict(states['scheduler'])
            epoch = states['epoch']
            best_score = states['best_score']
        else:
            epoch = 0
            best_score = 0.0
        
        return epoch, best_score
    
    def store_train_env(self, epoch, net, ema_net, optimizer, scheduler, score, best_score, old_path, last_path, best_path, mgpu):
        # extract last state
        last_states = {
            'net': net.module.state_dict() if mgpu else net.state_dict()
            'ema_net': ema_net.module.state_dict() if mgpu else ema_net.state_dict()
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_score': best_score,
            'epoch': epoch+1,
        }
        
        # update old result for backup
        if epoch > 0:
            if os.path.exists(old_path):
                os.remove(old_path)
            os.rename(last_path, old_path)

        # store last state
        torch.save(last_states, last_path)
        
        # store best result state
        if score > best_score:
            torch.save(last_states, best_path)
        return
    
    # Load test environment
    def load_test_env(self, ema_net, path, mgpu=False, user_set_devices=None):
        states = self.get_states(path, mgpu, user_set_devices)
        self._load_model(ema_net, states['ema_net'], mgpu)
        return
    
    # Get stored states 
    def get_states(self, path, mgpu=False, user_set_devices=None):
        map_location = {'cuda:%d'%user_set_devices[0]: 'cuda:%d'%user_set_devices[dist.get_rank()]} if mgpu else None
        return torch.load(path, map_location=map_location)
    
    # Define model
    def define_model(self, net, ngpus_per_node, virtual_device, device, master, mgpu, addr, port):
        if mgpu:
            torch.distributed.init_process_group(backend='gloo', store=dist.TCPStore(addr, port, ngpus_per_node, master), world_size=ngpus_per_node, rank=virtual_device)
            net = DistributedDataParallel(net.to(device), device_ids=device)
        else:
            net = net.to(device)
        return net
    
    # Load weights to model
    def load_model(self, net, weight, mgpu=False):
        if mgpu:
            net.module.load_state_dict(weight)
            dist.barrier()
        else:
            net.load_state_dict(weight)
        return
    
    # Check if a dataset is divisible by gpu device num
    def is_divisible(self, dataset, ngpus_per_node):
        return len(dataset) % ngpus_per_node == 0

    # Norm image
    def norm(self, img):
        return (img - 0.5) * 2.0

    # Recover imgae
    def denorm(self, img):
        return img / 2.0 + 0.5
