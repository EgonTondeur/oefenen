import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--animal_1", type=str)
parser.add_argument("--animal_2", type=str)
parser.add_argument("--animal_3", type=str)
parser.add_argument("--split", type=int)
parser.add_argument("--training_data", type=str)
parser.add_argument("--testing_data", type=str)
args = parser.parse_args()

train_dir = Path(args.training_data)
test_dir = Path(args.testing_data)
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

for label, folder in enumerate([args.animal_1, args.animal_2, args.animal_3]):
    images = list(Path(folder).glob("*.jpg"))
    random.shuffle(images)

    split_idx = int(len(images) * (100 - args.split) / 100)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        dst = train_dir / f"{label}_{img.name}"
        shutil.copy(img, dst)

    for img in test_imgs:
        dst = test_dir / f"{label}_{img.name}"
        shutil.copy(img, dst)
