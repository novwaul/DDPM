from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn

class Scores(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False) # use original Inception model and give [-1, 1] as a input
        
        self.results = dict()
        self.model.fc.register_forward_hook(self._get_result('IS', self.results))
        self.model.avgpool.register_forward_hook(self._get_result('FID', self.results))

        self.up = nn.UpSample(size=(299,299), mode='bilinear')
        self.softmax = nn.Softmax(dim=-1) # for IS

        self.eps = 1e-8 # to prevent log(0)

    def _get_result(self, score_name, container):
        def hook(model, input, output):
            container[score_name] = output.detach()
        return hook

    def forward(self, x, gt):
        with torch.no_grad():
            # get inception results
            B, *_ = x.shape
            z = torch.cat((x, gt)) # shape = (2*B, H, W, C); to calculate both at once
            z = self.upsample(z)
            _ = self.model(z)
            
            # calculate Inception Score
            fake, _ = torch.split(self.results['IS'].squeeze(), B) # get generated images onlyl; shape = (B, 1000) where B = num of x
            posteriors_yx = self.softmax(fake) # lisf of p(y|x); shape = (B, 1000)
            probability_y = torch.mean(posteriors_yx, dim=0) # marginal distribtion of y = p(y) = mean{ p(y|x) } on x; shape = (1000)
            entropy = torch.sum(posteriors_yx * torch.log(posteriors_yx + self.eps), dim=1) # sum{ p(y|x) * log( p(y|x) ) } on y; shape = (B)
            cross_entropy = torch.sum(posteriors_yx * torch.log(probability_y + self.eps), dim=1) # sum{ p(y|x) * log( p(y) ) } on y; shape = (B)
            log_sharpness = torch.mean(entropy, dim=0)
            log_diversity = -torch.mean(cross_entropy, dim=0)
            inception_score = torch.exp(log_sharpness + log_diversity) # sharpness x diversity
            
            # calculate FID Score
            fake, real = torch.split(self.results['FID'].squeeze(), B)
            mu_fake, mu_real = torch.mean(fake, axis=0), torch.mean(real, axis=0)
            sigma_fake, sigma_real = torch.cov(fake), torch.cov(real)
            fid_score = torch.square(mu_fake - mu_real) + torch.trace(sigma_fake + sigma_real - 2*torch.sqrt(sigma_fake*sigma_real))

        return inception_score, fid_score
