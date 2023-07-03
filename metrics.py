import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class Metrics(nn.Module):
    def __init__(self, device, ):
        super().__init__()
        self.device
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False) # use original Inception model and give [-1, 1] as a input
        self.model.eval()

        self.results = dict()
        self.model.fc.register_forward_hook(self._get_result('IS', self.results))
        self.model.avgpool.register_forward_hook(self._get_result('FID', self.results))

        self.up = nn.UpSample(size=(299,299), mode='bilinear')
        self.softmax = nn.Softmax(dim=-1) # for IS

        self.fid_activations = None
        self.is_activations = None
    
        self.eps = 1e-8 # to prevent log(0)

    def _get_result(self, score_name, container):
        def hook(model, input, output):
            container[score_name] = output.detach()
        return hook
    
    def _calc_fid_stats(self):
        act = self.fid_activations
        mean = torch.mean(act, axis=0)
        sigma = torch.cov(torch.cat((act.t(), act.t()), axis=0))
        return mean, sigma
    
    def _calc_fid(self):
        mean_gt, sigma_gt = self.mean_gt, self.sigma_gt
        mean_x0, sigma_x0 = self._calc_fid_stats()
        fid_score = torch.square(mean_gt-mean_x0) + torch.trace(sigma_gt + sigma_x0 - 2*torch.sqrt(sigma_gt*sigma_x0))
        return fid_score

    def _calc_is(self):
        scores = list()
        for acts in torch.split(self.is_activations, len(self.fid_activations)//10):
            posteriors_yx = self.softmax(acts) # lisf of p(y|x); as pytorch inception_v3 does not have softmax, apply softmax to output; shape = (B, 1000)
            probability_y = torch.mean(posteriors_yx, dim=0) # marginal distribtion of y = p(y) = mean{ p(y|x) } on x; shape = (1000)
            entropy = torch.sum(posteriors_yx * torch.log(posteriors_yx + self.eps), dim=1) # sum{ p(y|x) * log( p(y|x) ) } on y; shape = (B)
            cross_entropy = torch.sum(posteriors_yx * torch.log(probability_y + self.eps), dim=1) # sum{ p(y|x) * log( p(y) ) } on y; shape = (B)
            log_sharpness = torch.mean(entropy, dim=0)
            log_diversity = -torch.mean(cross_entropy, dim=0)
            inception_score = torch.exp(log_sharpness + log_diversity) # sharpness x diversity
            scores.append(inception_score)
        
        scores = torch.stack(scores, dim=0)

        return torch.mean(scores), torch.std(scores)
    
    def get_activations(self):
        return self.is_activations, self.fid_activations
    
    def set_activations(self, is_acts, fid_acts):
        self.is_activations = is_acts
        self.fid_activations = fid_acts
    
    def setup(self, eval_dataloader):
        for img in eval_dataloader:
            img = img.to(self.device)
            self.update(img)
        self.mean_gt, self.sigma_gt = self._calc_fid_stats()
        return

    def update(self, img):
        with torch.no_grad()
            fid_act = self.results['FID'].squeeze()
            is_act = self.results['IS'].squeeze()
            self.fid_activations = torch.cat(self.fid_activations, fid_act) if self.fid_activations != None else fid_act
            self.is_activations = torch.cat(self.is_activations, is_act) if self.is_activations != None else is_act
        return

    def forward(self):
        inception_score = self._calc_is()
        fid_score = self._calc_fid()
            
        return inception_score, fid_score