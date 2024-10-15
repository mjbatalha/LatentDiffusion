import numpy as np
import torch
import random
import yaml

from pathlib import Path
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
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
        
        self.writer = SummaryWriter(**self.tr_cfg["writter"])
        self.save_config(config)
        self.n_batch = None

    def train(self):

        self.n_batch = 0
        for epoch in tqdm(range(self.tr_cfg["n_epochs"]), desc="epochs"):

            imgs = make_grid(self.sample())
            self.writer.add_image(" || ".join(self.tr_cfg["sample_prompts"]), imgs, epoch)

            metrics = self.train_epoch()
            self.writer.add_scalar("epoch_loss", np.mean(metrics["batch_loss"]), epoch)
            self.writer.add_scalar("epoch_log_loss", np.mean(metrics["batch_log_loss"]), epoch)

        imgs = make_grid(self.sample())
        self.writer.add_image(" || ".join(self.tr_cfg["sample_prompts"]), imgs, epoch + 1)

        self.writer.close()

    def train_epoch(self):

        self.model.train()

        metrics = {"batch_loss": [], "batch_log_loss": []}
        for batch in tqdm(self.dl, desc="batches", leave=False):
            
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

            metrics["batch_loss"].append(loss.item())
            metrics["batch_log_loss"].append(np.log(loss.item()))

            self.writer.add_scalar("batch_loss", loss.item(), self.n_batch)
            self.writer.add_scalar("batch_log_loss", np.log(loss.item()), self.n_batch)
            self.n_batch += 1

        return metrics

    @torch.inference_mode
    def sample(self):

        self.model.eval()

        z_shape = self.tr_cfg["z_shape"]
        prompts = self.tr_cfg["sample_prompts"]
        z = torch.randn((len(prompts), *z_shape)).to(self.device)
        c = self.model.text_conditioning(prompts).to(self.device)
        for t in reversed(tqdm(range(self.model.n_steps), desc="sampling", leave=False)):
            t = torch.full((len(prompts),), t).long().to(self.device)
            z = self.model.sample(z, t, c)
        imgs = self.model.decode(z)

        return imgs
    
    def save_config(self, config: dict):
        cfg_path = Path(self.tr_cfg["writter"]["log_dir"]) / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":

    with open("train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = TrainTxt2Img(config)
    trainer.train()

