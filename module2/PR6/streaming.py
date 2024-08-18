import numpy as np
from PIL import Image
from streaming import MDSWriter, StreamingDataset
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def save_cifar10_to_local(local_cache: Path):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=str(local_cache), train=True, download=True, transform=transform)

    images_path = local_cache / 'images'
    labels_path = local_cache / 'labels'
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    for idx, (image, label) in enumerate(dataset):
        image_file = images_path / f"{idx}.png"
        label_file = labels_path / f"{idx}.txt"

        image = transforms.ToPILImage()(image)
        image.save(image_file)

        with open(label_file, 'w') as f:
            f.write(str(label))


def stream_cifar10_to_mds(local_cache: Path, path_to_save: Path):
    images_path = local_cache / 'images'
    labels_path = local_cache / 'labels'
    columns = {"image": "jpeg", "class": "int"}
    compression = "zstd"

    with MDSWriter(out=str(path_to_save), columns=columns, compression=compression) as out:
        for image_file in images_path.glob('*.png'):
            with open(labels_path / f"{image_file.stem}.txt") as f:
                label = int(f.read().strip())
            image = Image.open(image_file)
            sample = {
                "image": image,
                "class": label,
            }
            out.write(sample)


def get_dataloader(
        remote: str = "s3://datasets/random-data", local_cache: Path = Path("cache")
):
    dataset = StreamingDataset(local=str(local_cache), remote=remote, shuffle=True)
    print(dataset)
    sample = dataset[42]
    print(sample["image"], sample["class"])
    dataloader = DataLoader(dataset)
    print(f"PyTorch DataLoader = {dataloader}")


app = typer.Typer()
app.command()(save_cifar10_to_local)
app.command()(stream_cifar10_to_mds)
app.command()(get_dataloader)

if __name__ == "__main__":
    app()
