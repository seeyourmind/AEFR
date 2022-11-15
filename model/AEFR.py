import torch
import torch.nn as nn
import torch.nn.functional as F
import arch.models as models
import numpy as np


class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.latent_size = hyperparameters['latent_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.generalized = hyperparameters['generalized']
        self.reparameterize_with_noise = True

        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
            self.aux_data_size = 1024
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
            self.aux_data_size = 102
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10
            self.aux_data_size = 85
        elif self.DATASET == 'APY':
            self.num_classes = 32
            self.num_novel_classes = 12
            self.aux_data_size = 64
        elif self.DATASET == 'FLO':
            self.num_classes = 102
            self.num_novel_classes = 20
            self.aux_data_size = 1024

        feature_dimensions = [2048, self.aux_data_size]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}
        self.encoder[self.all_data_sources[0]] = models.GatedLinAttenUnit(feature_dimensions[0], self.latent_size, self.hidden_size_rule['encoder'], self.device)
        self.encoder[self.all_data_sources[1]] = models.normal_encoder_template(feature_dimensions[1], self.latent_size, self.hidden_size_rule['encoder'], self.device)

        self.decoder = models.decoder_template(self.latent_size, feature_dimensions[0], self.hidden_size_rule['decoder'], self.device)

        self.prehead0 = models.ProjectionHeadOld(self.latent_size, self.device)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def intra_mixup(self, mui, logvari, mua):
        # intra-Mixup
        alpha = 0.2
        logvara = torch.ones_like(mua)
        lam = np.random.beta(alpha, alpha)
        mu_mix = lam*mui + (1-lam)*mua
        logvar_mix = torch.log(lam**2*torch.exp(logvari) + (1-lam)**2*torch.exp(logvara))
        z_from_mix = self.reparameterize(mu_mix, logvar_mix)
        return z_from_mix

    def forward(self, img, att):
        # encoder inference
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, torch.ones_like(mu_att))
        
        z_from_mix = self.reparameterize(mu_att, logvar_img)
        # z_from_mix = self.intra_mixup(mu_img, logvar_img, mu_att)

        # decoder inference
        img_from_img = self.decoder(z_from_img)
        img_from_att = self.decoder(z_from_att)
        
        # SCC inference
        clf_from_img = self.prehead0(z_from_img)
        clf_from_att = self.prehead0(z_from_att)
        clf_from_mix = self.prehead0(z_from_mix)

        res = {'rec':  # reconstruction
                   {'img': img_from_img,
                    'imga': img_from_att,
                    },
               'dis':  # distribution
                   {'mui': mu_img,
                    'mua': mu_att,
                    'logvari': logvar_img,
                    },
               'cls':
                   {
                    'vis': clf_from_img,
                    'sem': clf_from_att,
                    'mix': clf_from_mix,
                   }
               }
        return res

