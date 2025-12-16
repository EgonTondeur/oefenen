import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--output_data", type=str)
args = parser.parse_args()

input_dir = Path(args.data)
output_dir = Path(args.output_data)
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img.save(output_dir / img_path.name)
