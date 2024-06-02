import os
import shutil
import pandas as pd
import requests
from zipfile import ZipFile
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NO_AUTH_URLS = {
    "dev": [
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partaa",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partab",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partac",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partad",
    ],
    "test": [
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip"
    ],
}

DOWNLOADED_DIRECTORY = "./DB/VoxCeleb1"
UNZIPPED_DIRECTORY = "./DB/VoxCeleb1/wav"

STRUCTURED_DIRECTORY = {
    "dev": "./DB/VoxCeleb1/dev_wav",
    "test": "./DB/VoxCeleb1/eval_wav",
}

META_FILE_PATH = "./DB/vox1_meta.csv"


def format_number(num):
    return f"{num:07d}"


def format_dataset_structure(metadata, dataset_type):
    logging.info(f"Formatting dataset structure for {dataset_type}")
    base_directory = UNZIPPED_DIRECTORY
    new_structured_directory = STRUCTURED_DIRECTORY[dataset_type]
    voxceleb_id_to_name = dict(zip(metadata["VoxCeleb1 ID"], metadata["VGGFace1 ID"]))

    for voxceleb_id in os.listdir(base_directory):
        if voxceleb_id not in voxceleb_id_to_name:
            logging.warning(f"VoxCeleb ID {voxceleb_id} not found in metadata")
            continue

        name = voxceleb_id_to_name[voxceleb_id]
        new_directory = os.path.join(new_structured_directory, name)
        os.makedirs(new_directory, exist_ok=True)

        current_directory = os.path.join(base_directory, voxceleb_id)
        for root, _, files in os.walk(current_directory):
            for file in files:
                if file.endswith(".wav"):
                    file_no = format_number(int(file.split(".")[0]))
                    new_filename = f"{os.path.basename(root)}_{file_no}.wav"
                    old_filepath = os.path.join(root, file)
                    new_filepath = os.path.join(new_directory, new_filename)
                    shutil.move(old_filepath, new_filepath)

    # for voxceleb_id in os.listdir(base_directory):
    #     old_directory = os.path.join(base_directory, voxceleb_id)
    #     if os.path.isdir(old_directory):
    #         shutil.rmtree(old_directory)


def download_file(url, save_path):
    logging.info(f"Downloading {url} to {save_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 Megabyte

    progress_bar = tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=url.split("/")[-1]
    )
    downloaded_size = 0

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=block_size):
            file.write(chunk)
            downloaded_size += len(chunk)
            progress_bar.update(len(chunk))
            progress_bar.set_description(
                f"{downloaded_size/1024/1024:.2f}MB/{total_size/1024/1024:.2f}MB"
            )

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        logging.error(f"ERROR: Download of {url} incomplete.")
    else:
        logging.info(f"Downloaded {url} to {save_path}")


def concatenate_files(urls, save_path):
    logging.info(f"Concatenating files to {DOWNLOADED_DIRECTORY}")
    concatenated_file_path = os.path.join(DOWNLOADED_DIRECTORY, "vox1_dev_wav.zip")
    with open(concatenated_file_path, "wb") as wfd:
        for url in urls:
            filename = url.split("/")[-1]
            part_path = os.path.join(save_path, filename)
            with open(part_path, "rb") as fd:
                shutil.copyfileobj(fd, wfd)
    logging.info(f"Concatenated files to {concatenated_file_path}")


def download_dataset(dataset_type):
    logging.info(f"Starting download for {dataset_type} dataset")
    urls = NO_AUTH_URLS[dataset_type]
    os.makedirs(DOWNLOADED_DIRECTORY, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        save_path = os.path.join(DOWNLOADED_DIRECTORY, filename)
        if os.path.exists(save_path):
            logging.info(f"Dataset {dataset_type} already downloaded!")
            continue
        download_file(url, save_path)

    if dataset_type == "dev":
        concatenate_files(urls, DOWNLOADED_DIRECTORY)
    logging.info(f"Completed download for {dataset_type} dataset")


def unzip_dataset(dataset_type):
    logging.info(f"Starting unzip for {dataset_type} dataset")
    for file in os.listdir(DOWNLOADED_DIRECTORY):
        if file.endswith(".zip") and file.find(dataset_type) != -1:
            file_path = os.path.join(DOWNLOADED_DIRECTORY, file)
            with ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(DOWNLOADED_DIRECTORY)
            # os.remove(file_path)
    logging.info(f"Completed unzip for {dataset_type} dataset")


if __name__ == "__main__":
    logging.info("Starting dataset processing")
    metadata = pd.read_csv(META_FILE_PATH, sep="\t")
    dataset_types = NO_AUTH_URLS.keys()
    for dataset_type in dataset_types:
        download_dataset(dataset_type)
        unzip_dataset(dataset_type)
        format_dataset_structure(metadata, dataset_type)
    logging.info("Completed dataset processing")
