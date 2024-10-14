import numpy as np
import torch
import random

from torch.nn import MSELoss
from torch.optim import Adam
from tqdm.auto import tqdm

from autoencoder import KLAutoencoder
from conditioning import CLIPTextEmbedder
from data import DiffusionDB
from ld_models import Text2ImgLDModel
from unet import UNet


LOSSES = {
    "mse": MSELoss,
}

OPTIMIZERS = {
    "adam": Adam,
}


class TrainTxt2Img(object):

    def __init__(self, config: dict):
        
        super(TrainTxt2Img, self).__init__()

        self.tr_cfg = config["train"].copy()
        self.device = self.tr_cfg["device"]

        torch.manual_seed(self.tr_cfg["seed"])
        np.random.seed(self.tr_cfg["seed"])
        random.seed(self.tr_cfg["seed"])

        vae = KLAutoencoder()
        text_embedder = CLIPTextEmbedder()
        unet = UNet(**config["unet"])
        dataset = DiffusionDB(**config["dataset"])
        dataset.apply_transforms()
        
        self.model = Text2ImgLDModel(unet, vae, text_embedder, **config["ld_model"]).to(self.device)
        self.ds = dataset.ds
        self.dl = dataset.get_dataloader()
        self.loss = [LOSSES[k](**v) for k, v in self.tr_cfg["loss"].items()][0]
        self.opt = [OPTIMIZERS[k](self.model.parameters(), **v) for k, v in self.tr_cfg["optimizer"].items()][0]

    def train(self):
        
        metrics = {
            "batch_loss": [],
            "epoch_loss": [],
            }
        
        for epoch in tqdm(range(self.tr_cfg["n_epochs"]), desc="epochs"):
            
            batch_metrics = self.train_epoch()
            metrics["batch_loss"].extend(batch_metrics["loss"])
            metrics["epoch_loss"].append(batch_metrics["loss"][-1])

    def train_epoch(self):

        self.model.train()

        batch_metrics = {"loss": []}
        for batch in tqdm(self.dl, desc="batches"):
            
            imgs = batch["image"].to(self.device)
            prompts = batch["prompt"]
            z = self.model.encode(imgs)
            t = torch.randint(0, self.model.n_steps, (imgs.shape[0],)).long().to(self.device)
            c = self.model.text_conditioning(prompts)
            
            z, noise = self.model.add_noise(z, t)
            noise_pred = self.model(z, t, c)
            loss = self.loss(noise, noise_pred)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            batch_metrics["loss"].append(loss.item())

        return batch_metrics

    @torch.inference_mode
    def eval_sample(self):

        self.model.eval()

        z_shape = self.tr_cfg["z_shape"]
        prompts = self.tr_cfg["eval_prompts"]
        z = torch.randn((len(prompts), *z_shape)).to(self.device)
        c = self.model.text_conditioning(prompts).to(self.device)
        
        for t in reversed(tqdm(range(self.model.n_steps), desc="sampling")):
            t = torch.full((len(prompts),), t).long().to(self.device)
            z = self.model.sample(z, t, c)


        imgs = self.model.decode(z)

        return imgs

