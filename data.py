from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from functools import partial

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


TRANSFORMS = {
    "resize": Resize,
    "totensor": ToTensor,
    "norm": Normalize,
}


class DiffusionDB(object):
    """

    see: https://poloclub.github.io/diffusiondb/
    
    """
    def __init__(self, config: dict, name: str = "poloclub/diffusiondb", size: str = "large_first_1k"): 
        super(DiffusionDB, self).__init__()

        self.config = config.copy()
        self.ds = load_dataset(name, size, split="train")        
        self.transforms = Compose([TRANSFORMS[k](**v) for k, v in self.config["transforms"].items()])

    def apply_transforms(self, batch_size: int = 100, n_threads: int = 10):
        self.ds = self.ds.map(
            self.transform_images, 
            batched=True, 
            batch_size=batch_size, 
            fn_kwargs={"n_threads": n_threads}
            )
        self.ds.set_format('torch', columns=['image', 'prompt'])

    def transform_images(self, batch, n_threads):
        transform_images_with_args = partial(lambda img: self.transforms(img))
        with ThreadPoolExecutor(max_workers=n_threads) as exec:
            batch["image"] = list(exec.map(transform_images_with_args, batch["image"]))
        return batch
    
    def get_dataloader(self):
        return DataLoader(self.ds, **self.config["dataloader"])
    
