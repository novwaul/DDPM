# DDPM
<p> Reimplementation of DDPM https://arxiv.org/abs/2006.11239 </p>

[2026.06.24 edit]
The below results were measured without self-attention due to a typo in the self-attention block. The typo is now fixed.

## How to run the code

First, set hyper-paramters and image paths at line 30~47 in <code> main.py </code>.

For example,

```
settings['steps']=2000
settings['sample_steps']=100
settings['iters']=500000
settings['lr']=1e-5
...
```

After setting those parameters and paths,

**Train:** <code> python main.py -m UNet -a [channel num] </code>

**Test:** <code> python main.py -m UNet -a [channel num] -t </code>


## Result

#### A. Settings
|Tag|Setting|
|:---:|:---:|
|Base Channel|64|
|Train Batch Size|128|
|Train Iterations|145.6K|
|Trian Data|CIFAR10|
|Train Beta Scehdule|Linear Schedule from 1e-4 to 2e-2|
|Sample Beta Schedule|Linear Schedule from 1e-4 to 2e-2|
|Train Steps|1000|
|Sample Steps|1000|

#### B. Scores
|Dataset|IS (Mean, Std.)|FID|
|:---:|:---:|:---:|
|CIFAR10|(7.741, 0.110)|18.337|

#### C. Samples
|Image|
|:---:|
|<img width="256" height="32" alt="imageData" src="https://github.com/user-attachments/assets/1e90c2d1-7397-4214-a791-8264ab06edd8" />|

