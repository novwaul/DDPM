import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module): # for CelabA-HQ; 256x256 pixels
    def __init__(self, C, steps, channel_expansions, emb_expansion=4, resblock_per_down_stage=2, attn_depth=None, drp_rate=0.0): # from the original code; set 0.1 for CIFAR10 and 0.0 for the others.
        super().__init__()
        self.emb = TimeEmbedding(steps, dim=C, exp=emb_expansion)
        self.conv1 = Conv2d(C, C, 3)

        depth = len(channel_expansions)
        last_depth = depth-1
        if attn_depth == None:
            attn_depth = last_depth - 1 # at 16 pixels; 256 -> 128 -> 64 -> 32 -> 16 (attention) -> 8 (attention; default) -> 16 (attention) -> 64 -> 128 -> 256

        resblock_per_up_stage = resblock_per_down_stage + 1 # to match block connections between up stage and down stage, where { Down_WideResBlock_1 -> Up_WideResBlock_1 }, { Down_WideResBlock_2 -> Up_WideResBlock_2 }, and { Down_Block -> Up_WideResBlock_3 }

        self.down = nn.ModuleList()
        for d in range(depth):
            prev_exp = channel_expansions[d-1] if d > 0 else 1
            exp = channel_expansions[d]
            
            for i in range(resblock_per_down_stage):
                res_block = WideResNetBlock((prev_exp if i == 0 else exp)*C, exp*C, emb_demension=emb_expansion*C, attention=(d == attn_depth), drp_rate=drp_rate)
                self.down.append(res_block)
            
            if d < last_depth:
                self.down.append(DownBlock(exp*C, exp*C))

        self.up = nn.ModuleList()
        for d in reversed(range(depth)):
            prev_exp =  channel_expansions[d] + (channel_expansions[d+1] if d < depth-1 else channel_expansions[-1])
            exp = channel_expansions[d]

            for i in range(resblock_per_up_stage):
                res_block = WideResNetBlock((prev_exp if i == resblock_per_up_stage-1 else exp)*C, exp*C, emb_demension=emb_expansion*C, attention=(d == attn_depth), drp_rate=drp_rate)
                self.up.append(res_block)
            
            if d > 0:
                self.up.append(UpBlock(exp*C, exp*C))

        self.mid = nn.Sequential(
            WideResNetBlock(channel_expansions[-1]*C, channel_expansions[-1]*C, emb_demension=emb_expansion*C, attention=True, drp_rate=drp_rate),
            WideResNetBlock(channel_expansions[-1]*C, channel_expansions[-1]*C, emb_demension=emb_expansion*C, attention=False, drp_rate=drp_rate)
        )

        self.gn = GroupNorm(channel_expansions[0]*C)
        self.silu = nn.SiLU()
        self.conv2 = Conv2d(channel_expansions[0]*C, C, kernel=3, gain=1e-10)
        
    def forward(self, x, times):
        emb = self.emb(times)

        connections = list()

        z = self.conv1(x)
        connections.append(z)

        for module in self.down:
            z = module(z, emb) if isinstance(module, WideResNetBlock) else module(z)
            connections.append(z)
        
        z = self.mid(z)
        
        for module in self.up:
            z = module(torch.cat((z, connections.pop()), dim=1), emb) if isinstance(module, WideResNetBlock) else module(z)
        
        z = self.gn(z)
        z = self.silu(z)
        out = self.conv2(z)

        return out

class TimeEmbedding(nn.Module):
    def __init__(self, steps, dim, exp):
        super().__init__()
        self.linear1 = Linear(dim, exp*dim)
        self.silu = nn.SiLU()
        self.linear2 = Linear(exp*dim, exp*dim)

        possible_times = torch.arange(steps, dtype=torch.float)
        x = torch.log(1e5) / (dim//2 - 1) # log( 10000^(1 / (d/2 - 1)) )
        x = torch.exp( torch.range(dim//2) * -x ) # 1 / 10000^(i / (d/2 - 1)) 
        x = possible_times.reshape((-1, 1)) * x.reshape((1, -1)) # t / 10000^(i / (d/2 - 1))
        possible_embs = torch.concat((torch.sin(x), torch.cos(x)), dim=1) # sin( t / 10000^(i / (d/2 - 1)) ) and cos( t / 10000^(i / (d/2 - 1)) )
        
        if dim % 2 != 0:
            possible_embs = F.pad(possible_embs, pad=(0, 1)) # add zero pad at last

        self.possible_embs = nn.Embedding.from_pretrained(possible_embs) # shape=(steps, dim)

    def forward(self, times):
        emb = self.possible_embs(times)
        emb = self.linear1(emb)
        emb = self.silu(emb)
        emb = self.linear2(emb)

        return emb # shape = (batch, exp*dim)

class WideResNetBlock(nn.Module): # DDPM ResBlock
    def __init__(self, in_channel, out_channel, emb_dimension, attention, drp_rate):
        super().__init__()
        self.do_attention = attention
        self.do_down = down
        self.do_up = up
        self.is_match = in_channel == out_channel
        self.C = out_channel

        self.gn1 = GroupNorm(in_channel)
        self.silu1 = nn.SiLU()
        self.conv1 = Conv2d(in_channel, out_channel, kernel=3)
        
        self.silu2 = nn.SiLU()
        self.linear1= Linear(emb_dimension, out_channel)

        self.gn2 = GroupNorm(out_channel)
        self.silu3 = nn.SiLU()
        self.dropout = nn.Dropout(drp_rate) 
        self.conv2 = Conv2d(out_channel, out_channel, kernel=3, gain=1e-10)

        if not self.is_match: # to match 'channel' betweem 'x' and 'z'
            self.linear2 = Linear(in_channel, out_channel)
        
        if self.do_attention:
            self.att = SelfAttentionBlock(out_channel) 

    def forward(x, emb):
        z = self.gn1(x)
        z = self.silu1(z)
        z = self.conv1(z)

        B, _, H, W = x.shape
        C = self.C
        emb = self.silu2(emb)
        z = self.linear1(emb).reshape(B, C, H, W) + z

        z = self.gn2(z)
        z = self.silu3(z)
        z = self.dropout(z) 
        z = self.conv2(z)

        if not self.is_match:
            x = x.permute(0, 2, 3, 1) # shape=(B,C,H,W) -> (B,H,W,C)
            x = self.linear2(x)
            x = x.permute(0, 3, 1, 2) # shape=(B,H,W,C) -> (B,C,H,W)

        out = x + z

        if self.do_attention:
            out = self.att(out)
        
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gn = GroupNorm(dim)
        self.qkv = Linear(dim, 3*dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = Linear(dim, dim, gain=1e-10)

    def forward(self, x):
        B, C, H, W = x.shape

        z = self.gn(x)
        z = z.permute(0, 2, 3, 1) # shape=(B,C,H,W) -> (B,H,W,C)
        qkv = self.qkv(z).view(B, H*W, 3, C).permute(2,0,1,3) # shape=(B,H,W,3*C) -> (3,B,H*W,C)
        q,k,v = qkv[0], qkv[1], qkv[2] # (B,H*W,C)

        z = torch.matmul(q, k.transpose(-2,-1)) / (C**0.5) # shape=(B,H*W,H*W)
        attention = self.softmax(z)
        self_attention = torch.matmul(attention, v) # shape=(B,H*W,C)

        z = self.proj(z).permute(0, 2, 1).reshape(B, C, H, W)
        return x + z

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv(in_channel, out_channel, kernel=3, stride=2)
    
    def forward(self, x): 
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv(in_channel, out_channel, kernel=3)
    
    def forward(self, x): 
        x = self.upsample(x)
        x = self.conv(x)
        return x

class GroupNorm(nn.Module):
    def __init__(self, in_channel, num_groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channel) # same as the TensorFlow default

    def forward(x):
        return self.group_norm(x)

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, gain=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=1)
        nn.init.xavier_uniform_(self.conv.weight, gain=torch.sqrt(gain)) # the original code initialization
        nn.init.constant_(self.conv.bias, 0.0) # the original code initialization
    
    def forward(x):
        return self.conv(x)

class Linear(nn.Module):
    def __init__(self, in_feature, out_feature, gain=1.0):
        super().__init__()
        self.linear = nn.Linear(in_feature, out_feature)
        nn.init.xavier_uniform_(self.linear.weight, gain=torch.sqrt(gain)) # the original code initialization
        nn.init.constant_(self.linear.bias, 0.0) # the original code initialization

    def forward(x):
        return self.linear(x)

