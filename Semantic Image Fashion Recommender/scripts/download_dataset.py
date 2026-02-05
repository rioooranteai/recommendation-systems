import shutil
from pathlib import Path
import kagglehub

script_dir = Path(__file__).parent
project_root = script_dir.parent

data_dir = project_root / "data"
fashion_data_dir = data_dir / "fashion-mini"

path = kagglehub.dataset_download("nirmalsankalana/mini-product-image-and-text-dataset")

if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

if fashion_data_dir.exists():
    shutil.rmtree(fashion_data_dir)

shutil.copytree(path, fashion_data_dir)
