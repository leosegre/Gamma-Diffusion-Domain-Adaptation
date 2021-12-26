import pytorch_lightning as pl
import torch
from models.diffusion import Model
import argparse
import numpy as np
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import get_dataset, data_transform, inverse_data_transform
from pytorch_lightning import Trainer
from main import dict2namespace
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image


class NGModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        betas = torch.linspace(self.hparams.beta_start, self.hparams.beta_end, self.hparams.n_iter)
        alphas = 1 - betas
        alpha_bar = alphas.cumprod(dim=0)
        self.alpha_sample = torch.cat([torch.FloatTensor([1]), alpha_bar[:-1]])
        #load pretrained
        with open(self.hparams.config_path, "r") as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        ddpm = Model(new_config)
        self.config = new_config
        self.ddpm = ddpm
        self.ddpm_betas = torch.linspace(self.hparams.beta_0, self.hparams.beta_N, steps=1000)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--grad_clip', type=float, default=1.0)
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--precision', type=int, default=32)
        parser.add_argument('--config_name', type=str, default='celebra')
        parser.add_argument('--exp', type=str, default='/media/data1/robinsr/img_data')
        parser.add_argument('--config_path', type=str, default='configs/celebra.yml')
        parser.add_argument('--ckpt_path', type=str, default='/media/data1/robinsr/model_img/ckpt.pth')
        parser.add_argument('--beta_0', type=float, default=0.0001)
        parser.add_argument('--beta_N', type=float, default=0.02)
        parser.add_argument('--N_sample_val', type=int, default=2)
        parser.add_argument('--n_iter_min', type=int, default=25)
        parser.add_argument('--n_iter_max', type=int, default=30)
        parser.add_argument('--type', type=str, default='quad')
        parser.add_argument('--NS', action='store_true')
        parser.add_argument('--eta', type=float, default=-1)
        parser.add_argument('--factor_poisson', type=float, default=1)
        parser.add_argument('--loss', type=str, default='L2')
        parser.add_argument('--noise_type', type=str, default="Poisson")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.ddpm.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def sample_alphas(self, batch_size):
        t = torch.randint(
            low=0, high=1000, size=(batch_size // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, 1000 - t - 1], dim=0)[:batch_size]
        alphas = self.alpha_sample.to(self.device)
        alpha_tens = alphas.index_select(0,t).view(-1, 1, 1, 1)
        return alpha_tens, t

    def construct_images(self, batch, alphas, noise_type):
        """ alphas is (bs, 1, 1, 1)"""
        if noise_type == 'Poisson':
            batch = data_transform(self.hparams.config, batch)
            rates = torch.ones(batch.size()).to(self.device)*self.hparams.factor_poisson * (1 - alphas)
            eps = (torch.poisson(rates) - rates)/(self.hparams.factor_poisson**0.5)
            y_out = alphas.sqrt()*batch + eps
        elif noise_type == "Gaussian":
            eps = torch.randn_like(batch)
            eps = (1-alphas).sqrt() * eps
            y_out = alphas.sqrt()*batch + eps
        elif noise_type == "Laplace":
            scale = (torch.ones(batch.size()) * (1 - alphas).sqrt()) / (2)**0.5
            mean = torch.zeros(batch.size())
            m = torch.distributions.Laplace(mean, scale)
            eps = m.sample()
            y_out = alphas.sqrt() * batch + eps
        else:
            raise RuntimeError("Noise Type {} is not implemented !".format(noise_type))
        return y_out, eps

    def training_step(self, batch, batch_idx):
        batch, _ = batch
        batch_size = batch.size(0)
        alphas, t = self.sample_alphas(batch_size)
        alphas = alphas.type_as(batch)
        y_N, eps = self.construct_images(batch, alphas, self.hparams.noise_type)
        output = self.ddpm(y_N, t.float())
        eps_hat = output / (1 - alphas).sqrt()
        loss = self.compute_loss(eps, eps_hat)
        self.log('train_loss', loss)
        self.metrics(eps, eps_hat)
        return loss

    def compute_loss(self, eps, eps_hat):
        if self.hparams.loss == "L2":
            return (eps - eps_hat).square().sum(dim=(1, 2, 3)).mean(dim=0)
        else:
            raise RuntimeError("Loss {} is not implemented".format(self.hparams.loss))

    def metrics(self, eps, eps_hat):
        L1 = torch.nn.L1Loss()(eps, eps_hat)
        MSE = torch.nn.MSELoss()(eps, eps_hat)
        self.log('L1', L1)
        self.log('MSE', MSE)

    def val_metrics(self, eps, eps_hat):
        L1 = torch.nn.L1Loss()(eps, eps_hat)
        MSE = torch.nn.MSELoss()(eps, eps_hat)
        self.log('val_L1', L1)
        self.log('val_MSE', MSE)

    def validation_step(self, batch, batch_idx):
        batch, _ = batch
        batch_size = batch.size(0)
        alphas, t = self.sample_alphas(batch_size)
        alphas = alphas.type_as(batch)
        y_N, eps = self.construct_images(batch, alphas, self.hparams.noise_type)
        output = self.ddpm(y_N, t.float())
        eps_hat = output / (1 - alphas).sqrt()
        val_loss = self.compute_loss(eps, eps_hat)
        val_loss = torch.tensor(val_loss.item())
        self.log('val_loss', val_loss)
        self.val_metrics(eps, eps_hat)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_val_loss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = NGModule.add_model_specific_args(parser)
    args = parser.parse_args()
    log_dir = args.log_dir
    logger = TensorBoardLogger(log_dir, name='NG_noise')
    checkpoint_path = os.path.join(log_dir, 'NG_noise', 'version_' + str(logger.version), 'checkpoints')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_top_k=1, monitor='avg_val_loss', mode='min')

    ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    trainer = Trainer(gpus=ngpus, logger=logger, checkpoint_callback=checkpoint, precision=args.precision,
                      gradient_clip_val=args.grad_clip)
    # parse config file
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    args.config = new_config
    args.n_iter = new_config.diffusion.num_diffusion_timesteps
    args.beta_start = new_config.diffusion.beta_start
    args.beta_end = new_config.diffusion.beta_end
    log_path = os.path.join(log_dir, 'NG_noise', 'version_' + str(logger.version))
    model = NGModule(args)

    dataset, test_dataset = get_dataset(args, new_config)
    train_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    test_loader =DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    trainer.fit(model, train_loader, test_loader)