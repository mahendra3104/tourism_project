from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Mahendra-ML/tourism-project"
repo_type = "dataset"

# Get token from environment variable
hf_token = os.environ.get('HF_TOKEN')

api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
    print(f"Repository '{repo_id}' created.")

api.upload_folder(
    folder_path="/content/tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=hf_token
)
print("Upload successful!")
