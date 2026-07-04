
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "SriSair/tourism-dataset"
repo_type = "dataset"

# Initialize API client using environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if dataset repository exists
try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Dataset '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating it...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=os.getenv("HF_TOKEN")
    )

    print(f"Dataset '{repo_id}' created successfully.")

# Upload all files inside data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully.")
