

### Result at 373 epochs (18.2% of total epochs)
|IS Mean|IS Std.|FID|
|:---:|:---:|:---:|
|7.741| 0.110| 18.337|


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

### 64x64 to 256x256 Model

#### A. Settings
|Tag|Setting|
|:---:|:---:|
|Base Channel|56|
|Train Batch Size|4|
|Train Iterations|500K|
|Trian Data|DIV2K Train Set + Flickr2K Train Set from 1001 to 2650 images|
|Validation Data|DIV2K Validation Set|
|Test Data|Flickr2K Train Set from 1 to 1000 images|
|Train Data Augmentation|Random Crop, Random Flip, Random Rotation|
|Test Data Augmentation|Centor Crop|
|Train Learning Rate Schedule|Cosine Annealing Schedule from 1e-5 to 1e-7|
|Train Beta Scehdule|Linear Schedule from 1e-4 to 0.005|
|Sample Gamma Schedule|Linear Schedule from 1e-4 to 0.1|
|Train Steps|1000|
|Sample Steps|100|

#### B. Scores
|Dataset|IS (Mean, Std.)|FID|
|:---:|:---:|:---:|
|CIFAR10|(7.741, 0.110)|18.337|

<p>Note that this model does not train on 256x256 to 1024x1024.</p>
<p>Inception Score shows low values as cropped images are hard to recognize as an object. As crop size increases, Inception Score also increases.</p>

#### C. Samples
<p>Note that the below LR images are upsampled images by using bicubic interpolation.</p>

##### Validation (64x64 to 256x256)
|Tag|Image|
|:---:|:---:|
|LR|![LR64_val](https://github.com/novwaul/SR3/assets/53179332/f7e3974f-d503-43d1-9a13-3fe4ee2e8d0c)|
|Sample|![Sample64_val](https://github.com/novwaul/SR3/assets/53179332/70dba161-3b20-472d-b4b5-0dcc0748d657)|
|HR|![HR64_val](https://github.com/novwaul/SR3/assets/53179332/ca2736aa-e350-4a81-bdf6-6abb8313a55d)|

##### Test (64x64 to 256x256)
|Tag|Image|
|:---:|:---:|
|LR|![LR64](https://github.com/novwaul/SR3/assets/53179332/656a7d4b-1925-42b8-b74b-698b13ec98ff)|
|Sample|![Sample64](https://github.com/novwaul/SR3/assets/53179332/5a922a74-2770-4b5c-8ca6-aeb2a7ddd3f7)|
|HR|![HR64](https://github.com/novwaul/SR3/assets/53179332/c8e53193-4c86-4caf-aa79-d9d314a5a9c3)|

##### Test (256x256 to 1024x1024)
|Tag|Image|
|:---:|:---:|
|LR|![LR256](https://github.com/novwaul/SR3/assets/53179332/41a1d329-9123-4e11-a03d-66d92a528241)|
|Sample|![Sample256](https://github.com/novwaul/SR3/assets/53179332/6cdccc42-5ba4-4294-bf16-8ecd11cca827)|
|HR|![HR256](https://github.com/novwaul/SR3/assets/53179332/6a13c426-79bd-4b5e-bc95-45ade689fff6)|
