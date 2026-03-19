import argparse
import subprocess
import os
import shutil
import tempfile
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace.")
    parser.add_argument("--local_dir", type=str, default="./data")
    parser.add_argument("--force_unzip", action="store_true",
                        help="Force re-unzipping even if images folder exists.")
    args = parser.parse_args()

    # repos = ["BoKelvin/SLAKE", "RadGenome/PMC-VQA"]
    # repo_to_zip = {
    #     "BoKelvin/SLAKE": ["imgs.zip"],
    #     "RadGenome/PMC-VQA": ["images.zip", "images_2.zip"],
    # }

    # Download data
    repo_dir = os.path.join(args.local_dir, "BoKelvin/SLAKE")
    snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset", local_dir=repo_dir)

    dest_folder = os.path.join(repo_dir, "images")
    zip_path = os.path.join(repo_dir, "imgs")
    tmp_dir = tempfile.mkdtemp(dir=repo_dir)
    
    if os.path.exists(dest_folder):
        if args.force_unzip:
            shutil.rmtree(dest_folder)

    # Unzip
    subprocess.run(["unzip", "-o", zip_path, "-d", tmp_dir], check=True)
    
    # Move to final folder
    out_folder = "imgs"
    out_folder_path = os.path.join(tmp_dir, out_folder)
    os.makedirs(dest_folder, exist_ok=True)
    for folder in os.listdir(out_folder_path):
        src_file = os.path.join(out_folder_path, folder, "source.jpg")
        dst_file = os.path.join(dest_folder, f"{folder}.jpg")

        if not (os.path.exists(src_file)):
            print(f"Skipping file {src_file}: does not exist")
            continue

        shutil.move(src_file, dst_file)
    
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()