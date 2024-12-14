import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import pdb
def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000,
                            noise_scale=0.1,
                            noise_min=0.0001,
                            noise_max=0.02
                            ) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    print("in get_named.. func: ")
    print("noise_scale: ", noise_scale)
    print("noise_min: ", noise_min)
    print("noise_max: ", noise_max)
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        #scale = 1000 / num_diffusion_timesteps
        #beta_start = scale * 0.0001
        #beta_end = scale * 0.02

        beta_start = noise_scale * noise_min
        beta_end = noise_scale * noise_max
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class GaussianDiffusion(nn.Module):
    def __init__(self, dtype:torch.dtype, model, betas:np.ndarray, w:float, v:float,
                 noise_scale, noise_min, noise_max, eta, timesteps,
                 device:torch.device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.device=device
        self.betas = torch.tensor(betas,dtype=self.dtype).to(self.device)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.noise_scale=noise_scale,
        self.noise_min=noise_min,
        self.noise_max=noise_max,
        self.eta = eta
        self.timesteps = timesteps
        self.device = device
        self.alphas = 1 - self.betas
        #pdb.set_trace()
        self.log_alphas = torch.log(self.alphas).to(self.device)
        
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0).to(self.device)
        self.alphas_bar = torch.exp(self.log_alphas_bar).to(self.device)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0).to(self.device)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev).to(self.device)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev).to(self.device)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas.to(self.device)
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas).to(self.device)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar.to(self.device)
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar).to(self.device)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar).to(self.device)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar).to(self.device)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0)).to(self.device)
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar).to(self.device)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar).to(self.device)
        # using \beta_t
        #self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0).to(self.device)
        # using \tilde\beta_t
        self.vars = self.tilde_betas
        self.coef1 = torch.exp(-self.log_sqrt_alphas).to(self.device)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar.to(self.device)
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)

        # Coupled diffusion modeling (xT)
        ele = torch.zeros((self.T, self.T), dtype=torch.float64)
        for t in range(self.T):
            for s in range(t+1):
                ele[t,s] = torch.exp(0.5*(torch.log(self.alphas_bar[t]) + torch.log(self.betas[s]) - torch.log(self.alphas_bar[s])))

        c_numerator = torch.sum(ele, dim=1)
        c_denominator = c_numerator[-1]

        self.const_c = c_denominator.to(self.device)
        self.ct = torch.exp(torch.log(c_numerator) - torch.log(c_denominator)).to(self.device)
        self.ct_prev = F.pad(self.ct[:-1], [1,0], 'constant', 0.0).to(self.device)
        self.ct_prev_tmp = F.pad(self.ct[:-1], [1,0], 'constant', 1.0).to(self.device)

        # reverse process of xT
        # part_1(cp1), part_2(cp2), part_3(cp3), (cp1 + cp2 + cp3) * x_T
        self.cp1 = torch.exp(
            torch.log(self.ct_prev_tmp) + torch.log(self.betas) + torch.log(0.5 * self.alphas)
            - self.log_one_minus_alphas_bar)
        self.cp1 = F.pad(self.cp1[1:], [1,0], 'constant', 0.0).to(self.device)

        alphas_minus_alphas_bar = self.alphas - self.alphas_bar
        alphas_minus_alphas_bar = F.pad(alphas_minus_alphas_bar[1:], [1,0], 'constant', 1.0).to(self.device)
        self.cp2 = - torch.exp(
            torch.log(alphas_minus_alphas_bar) + 0.5*torch.log(self.betas)
            -torch.log(self.const_c) - self.log_one_minus_alphas_bar)
        self.cp2 = F.pad(self.cp2[1:], [1,0], 'constant', 0.0).to(self.device)

        self.cp3 = - torch.exp(
            torch.log(self.betas) + torch.log(self.ct) -
            self.log_one_minus_alphas_bar).to(self.device)


        #self.coef1 = torch.exp(-self.log_sqrt_alphas).to(self.device)
        #self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar.to(self.device)
        self.coef3 = self.coef1 * (self.cp1 + self.cp2 + self.cp3)
        self.coef3 = self.coef3.to(self.device, dtype=torch.float32)
        self.const_c  = self.const_c.to(dtype=torch.float32)
        self.ct = self.ct.to(dtype=torch.float32)
        self.ct_prev = self.ct_prev.to(dtype=torch.float32)

        print("debug: \n")
        print("betas:", self.betas)
        print("ele: ", ele)
        print("self.ct: ", self.ct)
        print("self.ct_prev: ", self.ct_prev)
        print("self.const_c: ", self.const_c)
        print("self.cp1: ", self.cp1)
        print("self.cp2: ", self.cp2)
        print("self.cp3: ", self.cp3)
        print("self.alphas: ", self.alphas)
        print("slef.alphas_bar: ", self.alphas_bar)
        print("self.coef3: ", self.coef3)
            
                
    @staticmethod
    def _extract(coef:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]
        #pdb.set_trace()

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    #def q_sample(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #    """
    #    sample from q(x_t|x_0)
    #    """
    #    #pdb.set_trace()
    #    eps = torch.randn_like(x_0, requires_grad=False)
    #    return self._extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape) * x_0 \
    #        + self._extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape) * eps, eps

    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor, x_T:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        #pdb.set_trace()
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape) * x_0 \
            + self._extract(self.ct.to(self.device), t, x_0.shape) * F.dropout(x_T, p=0.5) \
            + self._extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var
    def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        #model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        #pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        #pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        pred_eps = pred_eps_cond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1.to(self.device), t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2.to(self.device), t = t, x_shape = x_t.shape) * eps


    def rec_predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor, x_T:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1.to(self.device), t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2.to(self.device), t = t, x_shape = x_t.shape) * eps + \
            self._extract(coef = self.coef3.to(self.device), t = t, x_shape = x_t.shape) * x_T


    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise

    def rec_p_mean_variance(self, x_t:torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        #cemb_shape = model_kwargs['cemb'].shape
        #pred_eps_cond = self.model(x_t, t, **model_kwargs)
        assert 'x_T' in model_kwargs, "noise(x_T) should be in model_kwargs"
        x_T = model_kwargs['x_T']
        cvt_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k != 'x_T':
                cvt_model_kwargs[k] = v

        pred_eps = self.model(x_t, t, **cvt_model_kwargs)
        #model_kwargs['cemb'] = torch.zeros(cemb_shape, device=self.device)
        #pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        #pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        #pred_eps = pred_eps_cond

        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum()   == 0, f"nan in tensor t   when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"

        #p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_mean = self.rec_predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps, x_T=x_T)
        if self.training:
            p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
            return p_mean, p_var
        else:
            return p_mean
            
        
    def rec_p_sample(self, x_t: torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        if self.training:
            mean, var = self.rec_p_mean_variance(x_t, t, **model_kwargs)
            assert torch.isnan(mean).int().sum() == 0, f"nan in the tensor mean when t = {t[0]}"
            assert torch.isnan(var).int().sum() == 0,  f"nan in the tensor var when t = {t[0]}"
            epsilon_noise = torch.randn_like(x_t)
            epsilon_noise[t <= 0] = 0
            return mean + torch.sqrt(var) * epsilon_noise
        else:
            mean = self.rec_p_mean_variance(x_t, t, **model_kwargs)
            assert torch.isnan(mean).int().sum() == 0, f"nan in the tensor mean when t = {t[0]}"
            return mean

    def rec_sample(self, x_0, **model_kwargs) -> torch.Tensor:
        #local_rank = get_rank()
        local_rank = 0
        #if local_rank == 0:
        #    print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        #x_t = torch.randn(shape, device = self.device)
        # q_sample
        # sch 1: go from self.T; sch 2: random go from [0, self.T)
        tlist = torch.ones([x_0.shape[0]], device = self.device) * (self.T - 1)
        #tlist = tlist.astype(
        if self.training:
            eps = torch.rand_like(x_0, requires_grad=False)
            #print("tlist: ", tlist, tlist.dtype)
            x_t = self._extract(self.sqrt_alphas_bar, tlist.type(dtype=torch.long), x_0.shape) * x_0 \
                + self._extract(self.sqrt_one_minus_alphas_bar, tlist.type(dtype=torch.long), x_0.shape) * eps
        else:  
            x_t = self._extract(self.sqrt_alphas_bar, tlist.type(dtype=torch.long), x_0.shape) * x_0

        # p sample:
        for _ in range(self.T):
            #tlist -= 1
            #x_t = p_saple(xxx)
            x_t = self.rec_p_sample(x_t, tlist, **model_kwargs)
            tlist -= 1
        return x_t

    def rec_backward(self, x_t, once=False, **model_kwargs) -> torch.Tensor:
        #local_rank = get_rank()
        local_rank = 0
        #if local_rank == 0:
        #    print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        # p sample:
        tlist = torch.ones([x_t.shape[0]], device = self.device) * (self.T - 1)
        x_T_dict = {"x_T": x_t}
        model_kwargs.update(x_T_dict)
        #print("model_kwargs:", model_kwargs)
        for _ in range(self.T):
            #tlist -= 1
            #x_t = p_saple(xxx)
            x_t = self.rec_p_sample(x_t, tlist, **model_kwargs)
            if once:
                return x_t
            tlist -= 1
        return x_t

    def compute_alpha(self, betas, t):
        betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
        a = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
        return a

    def compute_c(self, c, t):
        c = torch.cat([torch.zeros(1).to(c.device), c], dim=0)
        r = c.index_select(0, t+1).view(-1,1)
        return r

    def get_seq(self):
        ans = list(range(0, self.T, self.timesteps))
        return ans

    def generalized_steps(self, x_t, seq, **model_kwargs) -> torch.Tensor:
        seq = self.get_seq()

        n = x_t.size(0)
        seq_next = [-1] + list(seq[:-1])
        #x0_preds = []
        #xs  = [x]
        x_T = model_kwargs['x_T']
        cvt_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k != 'x_T':
                cvt_model_kwargs[k] = v        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x_t.device)
            next_t = (torch.ones(n) * j).to(x_t.device)

            at = self.compute_alpha(self.betas, t.long())
            at_next = self.compute_alpha(self.betas, next_t.long())

            ct = self.compute_c(self.ct, t.long())
            ct_next = self.compute_c(self.ct, next_t.long())

            et = self.model(x_t, t, **cvt_model_kwargs)

            #print("et, ct, x_t, at.shape:", et.shape, ct.shape, x_t.shape, at.shape)
            x0_t = (x_t - ct * x_T - et * (1 - at).sqrt()) / at.sqrt()

            c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / ( 1 - at)).sqrt()

            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et + ct_next * x_T
        return x_t
            
    
    def rec_backward_ddim(self, x_t, **model_kwargs) -> torch.Tensor:
        #local_rank = get_rank()
        local_rank = 0
        #if local_rank == 0:
        #    print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        # p sample:
        tlist = torch.ones([x_t.shape[0]], device = self.device) * (self.T - 1)
        x_T_dict = {"x_T": x_t}
        model_kwargs.update(x_T_dict)
        seq = self.get_seq()
        return self.generalized_steps(x_t, seq, **model_kwargs)
        #print("model_kwargs:", model_kwargs)
        #for _ in range(self.T):
        #    #tlist -= 1
        #    #x_t = p_saple(xxx)
        #    x_t = self.rec_p_sample(x_t, tlist, **model_kwargs)
        #    tlist -= 1
        #return x_t

    def sample(self, shape:tuple, **model_kwargs) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        #local_rank = get_rank()
        local_rank = 0
        #if local_rank == 0:
        #    print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        #for _ in tqdm(range(self.T),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
        for _ in tqdm(range(self.T),dynamic_ncols=True, disable=True):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = torch.clamp(x_t, -1, 1)
        #if local_rank == 0:
        #    print('ending sampling process...')
        return x_t
    
    def ddim_p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef = self.alphas_bar, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_bar_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma = eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps) / torch.sqrt(alphas_bar_t) + \
            coef_eps * pred_eps
        return p_mean, p_var
    
    def ddim_p_sample(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor: 
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t , t.type(dtype=torch.long), prevt.type(dtype=torch.long), eta, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def ddim_sample(self, shape:tuple, num_steps:int, eta:float, select:str, **model_kwargs) -> torch.Tensor:
        local_rank = get_rank()
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')
        
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in tqdm(range(num_steps),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t

    def rec_q_sample(self, x_0:torch.Tensor, t:torch.Tensor, **model_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        #pdb.set_trace()
        #if model_kwargs == None:
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape) * eps, eps
        #if 'noise' in model_kwargs:
        #    noise = model_kwargs['noise']
        #    eps = torch.randn_like(x_0, requires_grad=False)
        #    return self._extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape) * x_0 \
        #    + self._extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape) * (eps + noise), eps
    
    
    def trainloss(self, x_0:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        if 'x_T' not in model_kwargs:
            raise NotImplementedError
        else:
            x_T = model_kwargs['x_T']
        x_t, eps = self.q_sample(x_0, t, x_T = x_T)
        #pred_eps = self.model(x_t, t, **model_kwargs)
        #print("eps: ", eps.shape, pred_eps.shape)
        #cemb_shape = model_kwargs['cemb'].shape
        cvt_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k != 'x_T':
                cvt_model_kwargs[k] = v
                #print("v.shape:", v.shape)
        pred_eps = self.model(x_t, t, **cvt_model_kwargs)
        #model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        #pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        #pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        #pred_eps = pred_eps_cond
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss
    
