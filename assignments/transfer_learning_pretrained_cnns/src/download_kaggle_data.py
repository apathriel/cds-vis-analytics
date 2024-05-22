# Python inquirer api query implementation
# Handle unzipping the file
# add docstrings

import asyncio
import click
import json
import os
from pathlib import Path
import shutil
from typing import *

import aiofiles.os
from assignments.transfer_learning_pretrained_cnns.src.utilities.logger_utilities import get_logger
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

logger = get_logger(__name__)


class DirectoryManipulator:
    def __init__(self, data_path: Union[str, Path], dir_rename_val: str = "in") -> None:
        self.data_path: Union[str, Path] = data_path
        self.dir_target_name_val: str = dir_rename_val
        self._dataset_dir_title: str = ""

    async def check_directory_exists(self, dir_path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, Path(dir_path).exists)

    async def rename_dataset_folder(self) -> None:
        try:
            logger.info(f"Renaming dataset folder to '{self.dir_target_name_val}'...")
            await aiofiles.os.rename(self.dataset_dir_title, self.dir_target_name_val)
            logger.info(f"Dataset folder renamed to '{self.dir_target_name_val}'!")
        except FileExistsError:
            logger.error(
                f"A directory with the name {self.dir_target_name_val} already exists."
            )
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    async def create_directory(self, directory_path: Union[str, Path]) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, os.mkdir, directory_path)
            logger.info(f"Directory {directory_path} has been created.")
        except FileExistsError:
            logger.error(f"Directory {directory_path} already exists.")
        except PermissionError:
            logger.error(
                f"Permission denied. Cannot create directory {directory_path}."
            )
        except OSError as e:
            logger.error(
                f"An error occurred while trying to create directory {directory_path}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    async def move_to_initialized_container_folder(
        self, file_to_move: Union[str, Path]
    ) -> None:
        if not await self.check_directory_exists(self.dir_target_name_val):
            await self.create_directory(self.dir_target_name_val)
        try:
            logger.info(
                f"Moving file {file_to_move} to directory {self.dir_target_name_val}..."
            )
            await aiofiles.os.rename(
                Path(file_to_move),
                Path(self.dir_target_name_val) / Path(file_to_move),
            )
            logger.info(
                f"File {file_to_move} moved to directory {self.dir_target_name_val} successfully!"
            )
        except Exception as e:
            logger.error(
                f"An error occurred while trying to move file {file_to_move} to directory {self.dir_target_name_val}."
            )
            logger.info(e)

    async def delete_directory(self, directory_path: Union[str, Path]) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, shutil.rmtree, directory_path)
            logger.info(f"Directory {directory_path} has been deleted.")
        except FileNotFoundError:
            logger.error(f"Directory {directory_path} not found.")
        except PermissionError:
            logger.error(
                f"Permission denied. Cannot delete directory {directory_path}."
            )
        except NotADirectoryError:
            logger.error(f"The provided path {directory_path} is not a directory.")
        except OSError as e:
            logger.error(
                f"An error occurred while trying to delete directory {directory_path}: {str(e)}"
            )
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")

        async def rename_and_delete_duplicate(self):
            logger.info(f"Manipulating directory structure according to pre-defined instructions...")
            new_name = self.data_path / self.dir_target_name_val
            download_name_path = Path(self.download_name) / self.download_name
            try:
                await self.delete_directory(download_name_path)
                await self.rename_dataset_folder(self.dir_target_name_val)
            except Exception as e:
                logger.info(f"An error occurred: {e}")

    @property
    def dataset_dir_title(self) -> str:
        return self._dataset_dir_title

    @dataset_dir_title.setter
    def dataset_dir_title(self, value: str) -> None:
        self._dataset_dir_title = value


class KaggleCredentialsManager:
    def __init__(
        self,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        file_path: str = "kaggle.json",
    ) -> None:
        self._file_path: str = file_path
        self._username: Optional[str] = username
        self._api_key: Optional[str] = api_key

    def load_creds_from_json(self) -> None:
        logger.info("Attempting to load Kaggle credentials from JSON file...")
        try:
            with open(self._file_path, "r") as file:
                self.update_creds_from_dict(json.load(file))
                logger.info("Kaggle credentials loaded successfully!")
        except FileNotFoundError:
            logger.error("JSON file not found. Please check the file path.")
        except Exception as error:
            logger.error("An error occurred while loading the JSON file.")
            logger.info(error)

    def load_creds_from_env(self) -> None:
        self._username = os.getenv("KAGGLE_USERNAME")
        self._api_key = os.getenv("KAGGLE_KEY")

    def update_creds_from_dict(self, credentials: Dict[str, str]) -> None:
        self.username = credentials["username"]
        self.api_key = credentials["key"]

    def instantiate_environment_variables(self) -> None:
        logger.info("Instantiating environment variables...")
        if self._api_key and self._username:
            os.environ["KAGGLE_USERNAME"] = self._username
            os.environ["KAGGLE_KEY"] = self._api_key
            logger.info("Environment variables instantiated successfully!")
        else:
            logger.error(
                "No credentials provided. Please check your credentials and try again."
            )

    @property
    def username(self) -> Optional[str]:
        return self._username

    @username.setter
    def username(self, value: str) -> None:
        self._username = value

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._api_key = value


class KaggleDatasetManager:
    def __init__(
        self,
        dataset_url: str,
        data_path: str,
        dir_manager: DirectoryManipulator,
        dataset_dir_title: str = "",
        dir_rename_val: str = "in",
        dir_manipulation_type: str = "rename",
        force_download: bool = False,
    ) -> None:
        self.dataset_url = dataset_url
        self.dataset_url_slug = self.construct_dataset_url_slug()
        self.dataset_dir_title = (
            dataset_dir_title
            if dataset_dir_title
            else self.dataset_url_slug.split("/")[-1]
        )
        self.data_path = data_path
        self.dir_rename_val = dir_rename_val
        self.dir_manipulation_type = dir_manipulation_type
        self.dir_manager = dir_manager
        self.force_download = force_download
        self.kaggle_api = KaggleApi()

    def construct_dataset_url_slug(self) -> str:
        return "/".join(self.dataset_url.split("//")[1].split("/")[2:4])

    def authenticate_kaggle_api(self) -> None:
        logger.info("Authenticating Kaggle API...")
        try:
            self.kaggle_api.authenticate()
            logger.info("Kaggle API authentication successful!")
        except Exception as error:
            logger.error(
                f"Kaggle API authentication failed. Please check your credentials and try again: Error {error}"
            )

    def list_files_in_kaggle_dataset(self, verbose: bool = True) -> None:
        self.authenticate_kaggle_api()
        logger.info(f"Listing files in dataset '{self.dataset_url_slug}'...")
        try:
            dataset_files = kaggle.api.dataset_list_files(self.dataset_url_slug).files
            for file in dataset_files:
                file_info = f"File: {file.name:<20}"
                if verbose:
                    file_info += f" | Type: {file.fileType:<5} | Size: {file.size:<5} | \nDescription: {file.description}"
                logger.info(file_info)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def get_number_of_files_in_kaggle_dataset(self) -> int:
        self.authenticate_kaggle_api()
        try:
            dataset_files = kaggle.api.dataset_list_files(self.dataset_url_slug).files
            return len(dataset_files)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def get_single_file_kaggle_dataset_title(self) -> Optional[str]:
        self.authenticate_kaggle_api()
        try:
            dataset_files = kaggle.api.dataset_list_files(self.dataset_url_slug).files
            if len(dataset_files) > 1:
                logger.error(
                    f"Multiple files found in dataset '{self.dataset_url_slug}'."
                )
            else:
                logger.info(
                    f"Returning single file '{dataset_files[0]}' from dataset '{self.dataset_url_slug}'."
                )
                return dataset_files[0].name
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    async def download_kaggle_dataset(self) -> None:
        self.authenticate_kaggle_api()
        try:
            logger.info(f"Attempting to download dataset '{self.dataset_url_slug}'...")
            kaggle.api.dataset_download_files(
                self.dataset_url_slug,
                path=self.data_path,
                unzip=True,
                force=self.force_download,
                quiet=False,
            )
            logger.info(f"Dataset '{self.dataset_dir_title}' downloaded successfully!")

            if self.dir_manipulation_type == "rename":
                self.dir_manager.dataset_dir_title = self.dataset_dir_title
                await self.dir_manager.rename_dataset_folder()
            elif self.dir_manipulation_type == "parent_move":
                await self.dir_manager.move_to_initialized_container_folder(
                    self.get_single_file_kaggle_dataset_title()
                )
        except Exception as e:
            logger.error(f"An error occurred: {e}")


@click.command()
@click.option(
    "--dataset_url",
    "-u",
    prompt=True,
    type=str,
    help="URL of the desired Kaggle dataset to be downloaded.",
    required=True,
)
@click.option(
    "--data_path",
    "-d",
    default=Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
    help="Path to download the dataset.",
)
@click.option(
    "--dir_rename_val",
    "-r",
    default="in",
    help="Name of the directory to rename the dataset folder to.",
)
@click.option(
    "--dir_manipulation_type",
    "-m",
    type=str,
    default="None",
    help="Type of directory manipulation to perform. Options: 'rename', 'parent_move' or 'None'. Default: 'None'.",
)
def main(dataset_url, data_path, dir_rename_val, dir_manipulation_type):

    creds = KaggleCredentialsManager("kaggle.json")
    creds.load_creds_from_json()

    manager = DirectoryManipulator(data_path)
    downloader = KaggleDatasetManager(
        dataset_url=dataset_url,
        data_path=data_path,
        dir_manipulation_type=dir_manipulation_type,
        dir_manager=manager,
        dir_rename_val=dir_rename_val,
    )

    downloader.list_files_in_kaggle_dataset()
    asyncio.run(downloader.download_kaggle_dataset())


if __name__ == "__main__":
    main()
