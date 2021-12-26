import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import subprocess

import torchvision.utils as tvu
import pdb
dbstop = pdb.set_trace

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def gamma_steps(x, seq, model, b, theta_0=1, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        a = (1 - b).cumprod(dim=0)
        k = (b / a)/theta_0**2
        theta = (a.sqrt()*theta_0)
        k_bar = k.cumsum(dim=0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long()).to(x.device)
            atm1 = compute_alpha(betas, next_t.long()).to(x.device)
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')
            output = model(x, t.float())
            e = output
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)
            mean = mean_eps
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            concentration = torch.ones(x.size()).to(x.device) * k_bar[j]
            rates = torch.ones(x.size()).to(x.device) * theta[j]
            m = torch.distributions.Gamma(concentration, 1 / rates)
            eps = m.sample()
            eps = eps - concentration * rates
            eps = eps / (1.0 - a[j]).sqrt()
            sample = mean + mask * eps * torch.exp(0.5 * logvar)
            xs.append(sample.to('cpu'))
    return xs, x0_preds

def sample_gamma(model, device, dim, theta_0=1):
    seq = [i for i in range(1000)]
    b = torch.linspace(0.0001, 0.02, steps=1000)
    a = (1 - b).cumprod(dim=0)
    k = (b / a ) / theta_0**2
    theta = (a.sqrt() * theta_0)
    k_bar = k.cumsum(dim=0)
    concentration = torch.ones((1, 3, dim, dim)).to(device) * k_bar[-1]
    rates = torch.ones(concentration.size()).to(device) * theta[-1]
    m = torch.distributions.Gamma(concentration, 1 / rates)
    x = m.sample() - rates*concentration
    xs, x_0 = gamma_steps(x, seq, model, b.to(device), theta_0=theta_0)
    img = x_0[-1][0]
    return ((1 + img) * 0.5).permute(1, 2, 0)

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        text_file = open(self.args.log_path + "/fid_out.txt", "w")
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.to(self.device)
                x = data_transform(self.config, x)
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                if self.config.data.noise_type == "Gaussian":
                    e = torch.randn_like(x)
                    # antithetic sampling
                    loss = loss_registry[config.model.type](model, x, t, e, b)
                elif self.config.data.noise_type == "Poisson":
                    loss = loss_registry['poisson'](model, x, t, b)
                elif self.config.data.noise_type == "Laplace":
                    loss = loss_registry['laplace'](model, x, t, b)
                elif self.config.data.noise_type == "Gamma":
                    loss = loss_registry['gamma'](model, x, t, b, theta_0=self.config.data.theta_0)
                    if step % 5000 == 9:
                        img = sample_gamma(model, self.device, x.size(-1), theta_0=self.config.data.theta_0)
                        img = (1 + img)*0.5
                        tb_logger.add_image("Image gamma tmp", img, step, dataformats="HWC")
                elif self.config.data.noise_type == "Concat":
                    loss = loss_registry['concat'](model, x, t, b, theta_0=self.config.data.theta_0)
                elif self.config.data.noise_type == "2Gauss":
                    sigma = torch.linspace(self.config.data.sigma_start, self.config.data.sigma_end,
                                           steps=self.num_timesteps).to(self.device)
                    loss = loss_registry['2gauss'](model, x, t, b, sigma)
                elif self.config.data.noise_type == "2Gamma":
                    loss = loss_registry['2gamma'](model, x, t, b, self.config.data.theta_start,
                                                   self.config.data.theta_end, p=self.config.data.p)
                tb_logger.add_scalar("loss", loss, global_step=step)

                if not self.config.data.calc_fid:
                    logging.info(f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                data_start = time.time()

                #### calc FID
                self.args.timesteps_orig = self.args.timesteps
                model.eval()
                # if self.config.data.calc_fid and step > 50000 and step % self.config.training.snapshot_freq == 0:
                if self.config.data.calc_fid and step % self.config.training.snapshot_freq == 0:

                    # create folder
                    if not os.path.exists(self.args.image_folder):
                        os.makedirs(self.args.image_folder)

                    # erase files
                    filelist = [f for f in os.listdir(self.args.image_folder)]
                    for f in filelist:
                        os.remove(os.path.join(self.args.image_folder, f))
                    filelist = [f for f in os.listdir(self.args.image_folder)]
                    # print('num of files - ' + str(len(filelist)))

                    # run sampling
                    self.args.timesteps = 10
                    self.sample_fid(model)
                    print('step is - ' + str(step))
                    command = 'python -m pytorch_fid /private/home/eliyan/iclr2020/WaveGrad/10242020/WaveGrad/robin/16022021/BetaWavesImages/Imgrad/data/datasets/CIFAR-10-images/train/all '
                    command = command + self.args.image_folder
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                    process.wait()
                    # dbstop()
                    FID_SCORE = str(process.stdout.read())
                    print('process.stdout - ' + FID_SCORE)

                    text_file.write("epoch - " + str(epoch))
                    text_file.write('process.stdout - ' + FID_SCORE)
                    # text_file.close()
                    model.train()


    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            # model.eval()
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        # print(f"starting from image {img_id}")
        total_n_samples = 50000
        # print('DEBUG REMOVE THIS')
        # total_n_samples = 2048
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            # from functions.denoising import generalized_steps
            # from functions.denoising import generalized_steps, generalized_gamma
            from functions.denoising import generalized_steps, generalized_gamma, generalized_concat, generate_noise, generate_noise_2g, generalized_2g, generate_noise_2gamma, generalized_2gamma
            if self.config.data.noise_type == "Gamma":
                theta_0 = self.config.data.theta_0
                b = self.betas
                a = (1 - b).cumprod(dim=0)
                k = (b / a) / theta_0 ** 2
                theta = (a.sqrt() * theta_0)
                k_bar = k.cumsum(dim=0)
                concentration = torch.ones(x.size()).to(self.device) * k_bar[-1]
                rates = torch.ones(concentration.size()).to(self.device) * theta[-1]
                m = torch.distributions.Gamma(concentration, 1 / rates)
                x = m.sample() - rates * concentration
                xs = generalized_gamma(x, seq, model, self.betas, theta_0=theta_0, eta=self.args.eta)
            elif self.config.data.noise_type == "Concat":
                theta_0 = self.config.data.theta_0
                r = self.config.data.ratio
                b = self.betas
                a = (1 - b).cumprod(dim=0)
                k = (b / a) / theta_0 ** 2
                theta = (a.sqrt() * theta_0)
                k_bar = k.cumsum(dim=0)
                x = generate_noise(999, theta, k_bar, a, r, x, ntype="Concat")
                xs = generalized_concat(x, seq, model, self.betas, r, ntype="Concat", eta=self.args.eta)
            elif self.config.data.noise_type == "2Gauss":
                b = self.betas
                sigma = torch.linspace(self.config.data.sigma_start, self.config.data.sigma_start,
                                       steps=self.num_timesteps)
                x = generate_noise_2g(999, x, sigma)
                xs = generalized_2g(x, seq, model, b, sigma, eta=self.args.eta)
            elif self.config.data.noise_type == "2Gamma":
                theta_start, theta_end = self.config.data.theta_start, self.config.data.theta_end
                b = self.betas
                a = (1 - b).cumprod(dim=0)
                m = (theta_end - theta_start) / (a[-1] - a[0])
                c = theta_start - a[0] * m
                theta = a * m + c
                x = generate_noise_2gamma(999, x, theta)
                xs = generalized_2gamma(x, seq, model, b.to(x.device), theta, eta=0)
            else:
                xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs


            # xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            # x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
