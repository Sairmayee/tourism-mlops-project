
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "SriSair/tourism-streamlit-app"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if Space already exists
try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Space '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:

    print(f"Space '{repo_id}' not found. Creating it...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        space_sdk="docker",
        token=os.getenv("HF_TOKEN")
    )

    print(f"Space '{repo_id}' created successfully.")

# Upload deployment files
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Deployment files uploaded successfully.")
