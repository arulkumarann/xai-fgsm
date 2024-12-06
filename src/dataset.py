import os
import tarfile
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set it in the .env file.")

DATASET_URL = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/test_images.tar.gz"
DATA_DIR = "../data/imagenet"

def download_with_progress(url, token, destination_path):

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        raise ValueError(f"Failed to download dataset. HTTP Status Code: {response.status_code}. "
                         f"Message: {response.text}")

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  #1kb

    with open(destination_path, "wb") as file, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading", ncols=80
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def download_and_extract_dataset(url, download_dir, token):

    os.makedirs(download_dir, exist_ok=True)
    tar_file_path = os.path.join(download_dir, "test_images.tar.gz")

    if not os.path.exists(tar_file_path):
        print(f"Starting download from {url}...")
        download_with_progress(url, token, tar_file_path)
        print("Download complete.")

    if not tarfile.is_tarfile(tar_file_path):
        raise ValueError(f"The file {tar_file_path} is not a valid tar.gz archive.")

    extracted_dir = os.path.join(download_dir, "test_images")
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print("Extraction complete.")

    return extracted_dir

def get_imagenet_loader(data_dir, batch_size=32):

    dataset_dir = download_and_extract_dataset(DATASET_URL, data_dir, HUGGING_FACE_TOKEN)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
