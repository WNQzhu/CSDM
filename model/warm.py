import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import pickle as pkl

from .diffusion import GaussianDiffusion, get_named_beta_schedule
from .unet import Unet


class DropoutNet(nn.Module):

    def __init__(self, model: nn.Module, device, item_id_name='item_id'):
        super(DropoutNet, self).__init__()
        self.model = model
        self.item_id_name = item_id_name
        item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                            .repeat(item_emb.num_embeddings, 1)
        return

    def foward_without_itemid(self, xdict):
        bsz = xdict[self.item_id_name].shape[0]
        target = self.model.forward_with_item_id_emb(self.mean_item_emb.repeat([bsz, 1]), xdict)
        return target

    def foward(self, xdict):
        item_id_emb = xdict[self.item_id_name]
        target = self.model.forward_with_item_id_emb(item_id_emb, xdict)
        return target

class MetaE(nn.Module):
    
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(MetaE, self).__init__()
        self.build(model, warm_features, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        output_embedding_size = 0
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            type = self.model.description[item_f][1]
            if type == 'spr' or type == 'seq':
                output_embedding_size += emb_dim
            elif type == 'ctn':
                output_embedding_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 

        self.itemid_generator = nn.Sequential(
            nn.Linear(output_embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim),
        )
        return

    def init_metaE(self):
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_metaE(self):
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        # sideinfo_emb = torch.mean(torch.stack(item_embs, dim=1), dim=1)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        pred = self.itemid_generator(sideinfo_emb)
        return pred

    def forward(self, features_a, label_a, features_b, criterion=torch.nn.BCELoss(), lr=0.001):
        new_item_id_emb = self.warm_item_id(features_a)
        target_a = self.model.forward_with_item_id_emb(new_item_id_emb, features_a)
        loss_a = criterion(target_a, label_a.float())
        grad = autograd.grad(loss_a, new_item_id_emb, create_graph=True)[0]
        new_item_id_emb_update = new_item_id_emb - lr * grad
        target_b = self.model.forward_with_item_id_emb(new_item_id_emb_update, features_b)
        return loss_a, target_b

class MWUF(nn.Module):

    def __init__(self, 
                 model: nn.Module,
                 item_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(MWUF, self).__init__()
        self.build(model, item_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):

        self.model = model 
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            type = self.model.description[item_f][1]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        item_emb = self.model.emb_layer[self.item_id_name]
        new_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                            .repeat(item_emb.num_embeddings, 1)
        self.new_item_emb = nn.Embedding.from_pretrained(new_item_emb, \
                                                                freeze=False)
        # self.new_item_emb = nn.Embedding(item_emb.num_embeddings, item_emb.embedding_dim)
        self.meta_shift = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        self.meta_scale = nn.Sequential(
            nn.Linear(self.output_emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        self.get_item_avg_users_emb(train_loader, device)
        return

    def init_meta(self):
        for name, param in self.named_parameters():
            if ('meta_scale') in name or ('meta_shift' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_meta(self):
        for name, param in self.named_parameters():
            if ('meta_shift' in name) or ('meta_scale' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return
    
    def optimize_new_item_emb(self):
        for name, param in self.named_parameters():
            if 'new_item_emb' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def cold_forward(self, x_dict):
        item_id_emb = self.new_item_emb(x_dict[self.item_id_name])
        target = self.model.forward_with_item_id_emb(item_id_emb, x_dict)
        return target

    def warm_item_id(self, x_dict):
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.new_item_emb(item_ids).detach().squeeze()
        user_emb = self.avg_users_emb(item_ids).detach().squeeze()
        if user_emb.sum() == 0:
            user_emb = self.model.emb_layer['user_id'](x_dict['user_id']).squeeze()
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        item_emb = torch.concat(item_embs, dim=1).detach()
        # warm
        scale = self.meta_scale(item_emb)
        shift = self.meta_shift(user_emb)
        warm_item_id_emb = (scale * item_id_emb + shift)
        return warm_item_id_emb

    def forward(self, x_dict):
        warm_item_id_emb = self.warm_item_id(x_dict).unsqueeze(1)
        target = self.model.forward_with_item_id_emb(warm_item_id_emb, x_dict)
        return target

    def get_item_avg_users_emb(self, data_loaders, device):
        dataset_name = data_loaders.dataset.dataset_name
        path = "./datahub/item2users/{}_item2users.pkl".format(dataset_name)
        if os.path.exists(path):
            with open(path, 'rb+') as f:
                item2users = pkl.load(f)
        else:
            item2users = {}
            for features, _ in data_loaders:
                u_ids = features['user_id'].squeeze().tolist()
                i_ids = features['item_id'].squeeze().tolist()
                for i in range(len(i_ids)):
                    iid, uid = u_ids[i], i_ids[i]
                    if iid not in item2users.keys():
                        item2users[iid] = []
                    item2users[iid].append(uid)
            with open(path, 'wb+') as f:
                pkl.dump(item2users, f)
        avg_users_emb = []
        emb_dim = self.model.emb_layer[self.item_id_name].embedding_dim
        for item in range(self.model.emb_layer[self.item_id_name].num_embeddings):
            if item in item2users.keys():
                users = torch.Tensor(item2users[item]).long().to(device)
                avg_users_emb.append(self.model.emb_layer['user_id'](users).mean(dim=0))
            else:
                avg_users_emb.append(torch.zeros(emb_dim).to(device))
        avg_users_emb = torch.stack(avg_users_emb, dim=0) 
        self.avg_users_emb = nn.Embedding.from_pretrained(avg_users_emb, \
                                                                freeze=True)
        return

class CVAR(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(CVAR, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.log_v_encoder = nn.Linear(emb_dim, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.decoder = nn.Linear(17, 16)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        mean = self.mean_encoder(item_id_emb)
        log_v = self.log_v_encoder(item_id_emb)
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        freq = x_dict['count']
        pred = self.decoder(torch.concat([z, freq], 1))
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term

class CVAR(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(CVAR, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.log_v_encoder = nn.Linear(emb_dim, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.decoder = nn.Linear(17, 16)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        mean = self.mean_encoder(item_id_emb)
        log_v = self.log_v_encoder(item_id_emb)
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        freq = x_dict['count']
        pred = self.decoder(torch.concat([z, freq], 1))
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term

class CVAR(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(CVAR, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.log_v_encoder = nn.Linear(emb_dim, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.decoder = nn.Linear(17, 16)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        mean = self.mean_encoder(item_id_emb)
        log_v = self.log_v_encoder(item_id_emb)
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        freq = x_dict['count']
        pred = self.decoder(torch.concat([z, freq], 1))
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term



class DIFF(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16,
                 T = None,
                 w = None,
                 v = None,
                 noise_scale = None,
                 noise_min = None,
                 noise_max = None,
                 eta=None,
                 timesteps=None
                 ):
        super(DIFF, self).__init__()
        self.T = T
        self.w = w
        self.v = v
        self.noise_scale=noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.eta = eta
        self.timesteps = timesteps
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)

        betas = get_named_beta_schedule(num_diffusion_timesteps = self.T,
                                        noise_scale=self.noise_scale,
                                        noise_min=self.noise_min,
                                        noise_max=self.noise_max
                                        )
        self.unet = Unet(in_dims=[emb_dim, emb_dim, emb_dim],
                     out_dims=[emb_dim, emb_dim, emb_dim],
                     emb_size=2*emb_dim)
        
        self.diffusion = GaussianDiffusion(
            dtype=torch.float32,
            model=self.unet,
            betas = betas,
            w = self.w,
            v = self.v,
            noise_scale=self.noise_scale,
            noise_min = self.noise_min,
            noise_max = self.noise_max,
            eta = self.eta,
            timesteps=self.timesteps,
            device=device)

        print('self.T:', self.T)
        print('self.w:', self.w)
        print('self.v:', self.v)
        print('self.nosie_scale:', self.noise_scale)
        print('self.noise_min:', self.noise_min)
        print('self.noise_max:', self.noise_max)
        print('betas: ', betas)
        print('eta: ', eta)
        print('timesteps: ', timesteps)
        
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, emb_dim)
        #self.log_v_encoder = nn.Linear(emb_dim, 16)
        #self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size,  emb_dim)
        #self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        #self.decoder = nn.Linear(17, 16)
        self.decoder = nn.Linear(emb_dim + 1, emb_dim)

        #self.decoder = nn.Linear(emb_dim, emb_dim)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_diff(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_diff(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
            
        mean = self.mean_encoder(item_id_emb)
            
        sideinfo_emb = torch.concat(item_embs, dim=1)

        #bsz = sideinfo_emb.shape[0]
        #fake_time = torch.ones((bsz, 16), requires_grad=False).to(self.device)
        
        mean_p = self.mean_encoder_p(sideinfo_emb)
        
        reg_term = self.diffusion.trainloss(mean, x_T=mean_p)
        #reg_term = self.diffusion.trainloss(mean_p, x_T = mean)

        #z_p =  self.diffusion.rec_backward(x_t=mean_p)
        z_p = self.diffusion.rec_backward_ddim(x_t = mean_p)
        
        #log_v = self.log_v_encoder(item_id_emb)
        #mean_p = self.mean_encoder_p(sideinfo_emb)
        #log_v_p = self.log_v_encoder_p(sideinfo_emb)
        #reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        #reg_term = torch.square(mean - mean_p).sum()
        #reg_term = torch.zeros([1]).to(self.device)
        recon_term = torch.zeros([1]).to(self.device)
        #z = mean
        #z_p = mean_p
        #z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        #z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        freq = x_dict['count']

        #pred = self.decoder(z)
        #pred = self.decoder(torch.concat([z, freq], 1))
        #return pred, reg_term, recon_term
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        #recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term
