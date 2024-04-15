import asyncio
import json
import os
from pathlib import Path
import shutil

import aiofiles.os
import kaggle

def load_kaggle_credentials_from_json(file_path='kaggle.json'):
  try:
    with open(file_path, 'r') as file:
      return json.load(file)
  except Exception as error:
    print("[SYSTEM] An error occured while loading the Kaggle credentials.")
    print(error)

def instantiate_environment_variables(credentials=None):
  if credentials:
     os.environ["KAGGLE_USERNAME"] = credentials['username']
     os.environ["KAGGLE_KEY"] = credentials['key']
     print("[SYSTEM] Environment variables instantiated successfully!")
  else:
    print("[SYSTEM] No credentials provided. Please check your credentials and try again.")

def authenticate_kaggle_api():
  try:
    kaggle.api.authenticate()
    print("[SYSTEM] Kaggle API authentication successful!")
  except Exception as error:
    print("[SYSTEM] Kaggle API authentication failed. Please check your credentials and try again.")
    print(error)

def validate_data_folder_structure(dir_to_check: Path) -> bool: 
  if dir_to_check.is_dir():
    return False
  else:
     return True
    
async def rename_dataset_folder(old_name, new_name):
    try:
        await aiofiles.os.rename(old_name, new_name)
        print(f"[SYSTEM] Dataset folder renamed to '{new_name}'!")
    except FileExistsError:
        print(f"[SYSTEM] A directory with the name {new_name} already exists.")
        # Handle the error (e.g., delete the existing directory or choose a different name)
    except Exception as e:
        print(f"[SYSTEM] An error occurred: {e}")

async def delete_directory(directory_path):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, shutil.rmtree, directory_path)
        print(f"[SYSTEM] Directory {directory_path} has been deleted.")
    except FileNotFoundError:
        print(f"[SYSTEM] Directory {directory_path} not found.")
    except PermissionError:
        print(f"[SYSTEM] Permission denied. Cannot delete directory {directory_path}.")
    except NotADirectoryError:
        print(f"[SYSTEM] The provided path {directory_path} is not a directory.")
    except OSError as e:
        print(f"[SYSTEM] An error occurred while trying to delete directory {directory_path}: {str(e)}")
    except Exception as e:
        print(f"[SYSTEM] An unexpected error occurred: {e}")

async def manipulate_dir_structure(data_path, download_name, new_dir_name: str):
    print(f"[SYSTEM] Manipulating directory structure according to pre-defined instructions...")
    new_name = data_path / new_dir_name
    download_name_path = Path(download_name) / download_name
    try:
        await delete_directory(download_name_path)
        await rename_dataset_folder(download_name, new_name)
    except Exception as e:
        print(f"[SYSTEM] An error occurred: {e}")
        
async def download_kaggle_dataset(dataset_name, download_name: str, data_path: Path, dir_rename_val: str = 'in', manipulate_data: bool = True):
    if validate_data_folder_structure(data_path / dir_rename_val):
      try:
          print(f"[SYSTEM] Attempting to download dataset '{dataset_name}'...")
          kaggle.api.dataset_download_files(dataset_name, path=data_path, unzip=True)
          print(f"[SYSTEM] Dataset '{dataset_name}' downloaded successfully!")
          
          if manipulate_data:
              await manipulate_dir_structure(data_path, download_name, dir_rename_val)
        
      except Exception as e:
          print(f"[SYSTEM] An error occurred: {e}")
    else:
      print(f"[SYSTEM] Directory {data_path / dir_rename_val} already exists. Skipping download.")

if __name__ == '__main__':
    kaggle_dataset = 'patrickaudriaz/tobacco3482jpg'
    dataset_zip_name = 'Tobacco3482-jpg'
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    target_dir = Path(dir_path)
    credentials = load_kaggle_credentials_from_json('kaggle.json')

    instantiate_environment_variables(credentials)
    authenticate_kaggle_api()
    asyncio.run(download_kaggle_dataset(kaggle_dataset, dataset_zip_name, target_dir, manipulate_data=True))