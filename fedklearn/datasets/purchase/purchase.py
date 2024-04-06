import json
import os
import shutil
import logging
import time

import requests
import tarfile
from sklearn.model_selection import train_test_split


from fedklearn.datasets.purchase.constants import *
import torch
import pandas as pd
from torch.utils.data import Dataset

import numpy as np

class FederatedPurchaseDataset:
    """A class representing a federated dataset derived from the Adult dataset.

    This dataset is designed for federated learning scenarios where the data is split across multiple clients,
    and each client represents a specific task.

    Args:
        cache_dir (str): The directory path for caching downloaded and preprocessed data. Default is "./".

        download (bool): Whether to download the data if it is not found in the cache directory. Default is True.

        rng (np.random.Generator, optional):  An instance of a random number generator.

            If `None`, a new generator will be created. Default is `None`.

        force_generation (bool): Whether to force the generation of the dataset even if it already exists.
         Default is True.

         n_tasks (int): The number of tasks to split the data into. Default is 4.

         n_task_samples (int): The number of samples per task. Default is 1000.

         split_criterion (str): Criterion to use for splitting the data into tasks. Default is "random".

            test_frac (float): Fraction of the data to use for testing. Default is None.

         Attributes:

             raw_data_dir (str): The directory path for the raw data.

             intermediate_data_dir (str): The directory path for the preprocessed data.

             tasks_folder (str): The directory path for the tasks.

             _metadata_path (str): The file path for the metadata.

             _split_criterion_path (str): The file path to save the split criterion.

            task_id_to_name (dict): A dictionary mapping task IDs to task names.

            n_tasks (int): The number of tasks.

            n_tasks_samples (int): The number of samples per task.

            split_criterion (str): The criterion used for splitting the data into tasks.

            rng (np.random.Generator): An instance of a random number generator.
            If `None`, a new generator will be created..

            download (bool): Whether to download the data if it is not found in the cache directory. Default is True.

            force_generation (bool): Whether to force the generation of the dataset even if it already exists.

            cache_dir (str): The directory path for caching downloaded and preprocessed data. Default is "./".

            test_frac (float): Fraction of the data to use for testing. Default is None.


        Methods:

            _download_file(url, file_path): Downloads a file from a URL and saves it to a file path.

            _unzip_file(tgzfile_path, dest_dir): Unzips a .tgz file to a destination directory.

            _download_data(self): Downloads the raw data and unzips it to the raw data directory.

            _save_task_mapping(self, metadata_dict): Saves the task mapping to the metadata file.

            _load_task_mapping(self): Loads the task mapping from the metadata file.

            _preprocess(self): Preprocesses the raw data and saves it to the intermediate data directory.

            _iid_divide(self, df): Split a dataframe into a dictionary of dataframes.

            iid_tasks_divide(self, df): Divides the data in training and test dictionaries.

            _random_tasks_split(self, df): Splits the data into tasks using random sampling.

            _split_data_into_tasks(self, all_data): Splits the data into tasks using a specified criterion.

            get_task_dataset(self, task_id, mode='train'):
            Returns an instance of the `PurchaseDataset` class for a specific task and data split type.
            task_number: The task number or name.
            mode: The type of data split, either 'train' or 'test'. Default is 'train'

             """
    def __init__(self, cache_dir="./", download=True, rng=None, force_generation=True, n_tasks=4,
                 n_task_samples=1000, split_criterion="random", test_frac=None):
        self.cache_dir = cache_dir
        self.download = download
        self.force_generation = force_generation
        self.rng = rng if rng is not None else np.random.default_rng()
        self.split_criterion = split_criterion
        self.test_frac = test_frac

        self.n_tasks = n_tasks
        self.n_tasks_samples = n_task_samples

        self.raw_data_dir = os.path.join(self.cache_dir, "raw")
        self.intermediate_data_dir = os.path.join(self.cache_dir, "intermediate")
        self.tasks_folder = os.path.join(self.cache_dir, "tasks")

        self._metadata_path = os.path.join(self.cache_dir, "metadata.json")
        self._split_criterion_path = os.path.join(self.cache_dir, "split_criterion.json")

        if os.path.exists(self.tasks_folder) and not self.force_generation:
            logging.info("Processed data folders found. Loading existing files..")
            self._load_task_mapping()

        elif not self.download and self.force_generation:
            logging.info("Data found in the cache directory. Splitting data into tasks..")
            all_data = pd.read_csv(os.path.join(self.intermediate_data_dir, "dataset_purchase.csv"))
            self._generate_tasks(all_data)

        elif  not self.download:
            raise RuntimeError(
                f"Data is not found in {self.raw_data_dir}. Please set `download=True`."
            )
        else:

            # remove the task folder if it exists to avoid inconsistencies
            if os.path.exists(self.tasks_folder):
                shutil.rmtree(self.tasks_folder)

            logging.info("Downloading raw data..")
            os.makedirs(self.raw_data_dir, exist_ok=True)
            self._download_data()
            logging.info("Download complete. Processing data..")
            os.makedirs(self.intermediate_data_dir, exist_ok=True)

            all_data = self._preprocess()
            self._generate_tasks(all_data)


    def _generate_tasks(self, all_data):
        """
        Splits the data into tasks and saves the data to the tasks folder.
        Args:
            all_data(pd.DataFrame): a DataFrame containing the entire dataset.

        Returns:
        """

        train_tasks_dict, test_tasks_dict = self._split_data_into_tasks(all_data)
        task_dicts = [train_tasks_dict, test_tasks_dict]

        self.task_id_to_name = {f"{i}": task_name for i, task_name in enumerate(train_tasks_dict.keys())}

        for mode, task_dict in zip(['train', 'test'], task_dicts):
            for task_name, task_data in task_dict.items():
                task_cache_dir = os.path.join(self.cache_dir, 'tasks', self.split_criterion, task_name)
                os.makedirs(task_cache_dir, exist_ok=True)

                file_path = os.path.join(task_cache_dir, f'{mode}.csv')
                task_data.to_csv(file_path, index=False)

                logging.debug(f"{mode.capitalize()} data for task '{task_name}' cached at: {file_path}")

        self._save_task_mapping(self.task_id_to_name)

        self._save_split_criterion()


    def _save_split_criterion(self):
        """
        Saves the split criterion to a file.
        Returns:

        """
        with open(self._split_criterion_path, "w") as f:
            criterion_dict = {'split_criterion': self.split_criterion, 'n_tasks': self.n_tasks,
                              'n_task_samples': self.n_tasks_samples}
            json.dump(criterion_dict, f)


    @staticmethod
    def _download_file(url, file_path):
        """
        Downloads a file from a URL and saves it to a file path.
        Args:
            url(str): The URL of the file to download.
            file_path(os.Path): The file path to save the downloaded file.

        Returns:

        """
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    @staticmethod
    def _unzip_file(tgzfile_path, dest_dir):
        """
        Unzips a .tgz file to a destination directory.
        Args:
            tgzfile_path(str): The path to the .tgz file to unzip.
            dest_dir(str): The destination directory to extract the files to.

        Returns:

        """
        try:
            with tarfile.open(tgzfile_path, "r:gz") as tar:
                tar.extractall(dest_dir)
                logging.debug(f"Unzipped {tgzfile_path} to {dest_dir}.")
        except tarfile.ReadError:
            logging.error(f"Failed to extract {tgzfile_path}.")


    def _download_data(self):
        """
        Downloads the raw data and unzips it to the raw data directory.
        """
        tgzfile_path = os.path.join(self.raw_data_dir, TGZ_FILENAME)
        self._download_file(URL, tgzfile_path)
        logging.info(f"Data downloaded to {tgzfile_path}.")

        self._unzip_file(tgzfile_path, self.raw_data_dir)
        logging.info(f"Raw data unzipped to {self.raw_data_dir}.")


    def _save_task_mapping(self, metadata_dict):
        """
        Saves the task mapping to the metadata file.
        Args:
            metadata_dict(dict): A dictionary mapping task IDs to task names.

        Returns:
        """
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r") as f:
                metadata = json.load(f)
                metadata[self.split_criterion] = metadata_dict
            with open(self._metadata_path, "w") as f:
                json.dump(metadata, f)
        else:
            with open(self._metadata_path, "w") as f:
                metadata = {self.split_criterion: metadata_dict}
                json.dump(metadata, f)


    def _load_task_mapping(self):
        """
        Loads the task mapping from the metadata file.
        Returns:

        """
        with (open(self._metadata_path, "r") as f):
            metadata = json.load(f)
            self.task_id_to_name = metadata[self.split_criterion]


    def _preprocess(self):
        """
        Rename the columns to item IDs and save the processed data to the intermediate data folder.
        """
        df = pd.read_csv(os.path.join(self.raw_data_dir, FILENAME))
        item_ids = [f"{i}" for i in range(len(df.columns) - 1)]
        new_columns = ["class"] + item_ids
        df.columns = new_columns
        df['class'] = df['class'] - 1
        df.to_csv(os.path.join(self.intermediate_data_dir, "dataset_purchase.csv"), index=False)
        return df


    def _iid_tasks_divide(self, df):
        """
        Divides the data in training and test dictionaries in an iid fashion.
        Args:
            df(pd.DataFrame): The DataFrame containing the data to split.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing the training and test dictionaries.

        """

        if self.test_frac is  None:
            train_size = self.n_tasks_samples * self.n_tasks
        else:
            train_size = int(len(df) / self.n_tasks * (1 - self.test_frac))

        df_train = df.iloc[:train_size, :]
        df_test = df.iloc[train_size:, :]
        tasks_dict_train = self._iid_divide(df_train)
        tasks_dict_test = self._iid_divide(df_test)

        return tasks_dict_train, tasks_dict_test


    def _iid_divide(self, df):
        """
        Split a dataframe into a dictionary of dataframes.
        Args:
            df(pd.DataFrame): DataFrame to split into tasks.

        Returns:
            tasks_dict(Dict[str, pd.DataFrame]): A dictionary mapping task IDs to dataframes.

        """
        num_elems = len(df)
        group_size = int(len(df) // self.n_tasks)
        num_big_groups = num_elems - self.n_tasks * group_size
        num_small_groups = self.n_tasks - num_big_groups
        tasks_dict = dict()

        for i in range(num_small_groups):
            tasks_dict[f"{i}"] = df.iloc[group_size * i: group_size * (i + 1)]
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            tasks_dict[f"{i + num_small_groups}"] = df.iloc[bi + group_size * i:bi + group_size * (i + 1)]

        return tasks_dict


    def _random_tasks_split(self, df):
        """
        Splits the data into tasks using random sampling.
        Args:
            df(pd.DataFrame): The DataFrame containing the data to split.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing the training and test dictionaries.

        """
        shuffled_indices = self.rng.permutation(len(df))
        all_data = df.iloc[shuffled_indices]
        train_tasks_dict, test_tasks_dict = self._iid_tasks_divide(all_data)
        return train_tasks_dict, test_tasks_dict


    def _class_tasks_split(self, df):
        """
        Splits the data into tasks using the class labels.
        Args:
            df(pd.DataFrame):  The DataFrame containing the data to split.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing the training and test dictionaries.

        """

        assert df['class'].nunique() == self.n_tasks, "Number of tasks must be equal to the number of classes."

        min_n_samples = min(df['class'].value_counts())
        if self.n_tasks_samples is not None and self.n_tasks_samples >= min_n_samples:
            raise ValueError(f"Number of samples per task must be lower than {min_n_samples}.")

        train_tasks_dict = dict()
        test_task_dict = dict()

        for label in sorted(df['class'].unique()):
            task_dict = df[df['class'] == label]
            if self.test_frac is not None and self.n_tasks_samples is not None:
                logging.info("Both 'test_frac' and 'n_tasks_samples' are defined. Using 'test_frac' to split the data.")
            if self.test_frac is not None:
                train_tasks_dict[f"{label}"], test_task_dict[f"{label}"] = (
                    train_test_split(task_dict,test_size=self.test_frac, random_state=self.rng.integers(low=0, high=1000))
                )
            else:
                if self.n_tasks_samples is None:
                    raise ValueError("Number of samples or test fraction per task are not defined.")
                else:
                    train_tasks_dict[f"{label}"] = task_dict.sample(n=self.n_tasks_samples, random_state=self.rng)
                    test_task_dict[f"{label}"] = task_dict.drop(train_tasks_dict[f"{label}"].index)

        return train_tasks_dict, test_task_dict

    def _split_data_into_tasks(self, all_data):
        """
        Splits the data into tasks using a specified criterion. Available criteria are 'random' and 'class'.
        Args:
            all_data(pd.DataFrame): The DataFrame containing the data to split.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing the training and test dictionaries.
        """
        if self.tasks_folder is not None:
            os.makedirs(self.tasks_folder, exist_ok=True)
        else:
            raise ValueError("Tasks folder is not defined.")

        if self.n_tasks_samples is not None and self.n_tasks * self.n_tasks_samples > len(all_data):
            raise ValueError("The product between 'n_tasks' and 'n_tasks_samples' exceeds the number of available samples.")

        if self.split_criterion == "random":
            train_tasks_dict, test_tasks_dict = self._random_tasks_split(all_data)
        elif self.split_criterion == "class":
            train_tasks_dict, test_tasks_dict = self._class_tasks_split(all_data)

        else:
            raise ValueError(f"Split criterion '{self.split_criterion}' is not recognized.")

        return train_tasks_dict, test_tasks_dict


    def get_task_dataset(self, task_id, mode='train'):
        """
        Returns an instance of the `PurchaseDataset` class for a specific task and data split type.
        Args:
            task_id(str): The task number or name.
            mode(str): The type of data split, either 'train' or 'test'. Default is 'train'

        Returns:
            PurchaseDataset: An instance of the `PurchaseDataset` class.
        """

        task_id = str(task_id)

        if mode not in {'train', 'test'}:
            raise ValueError(f"Mode '{mode}' is not recognized.  Supported values are 'train' or 'test'.")

        task_name = self.task_id_to_name[task_id]
        task_cache_dir = os.path.join(self.cache_dir, 'tasks', self.split_criterion, task_name)
        file_path = os.path.join(task_cache_dir, f'{mode}.csv')
        task_data = pd.read_csv(file_path)

        return PurchaseDataset(task_data, name=task_name)


class PurchaseDataset(Dataset):
    def __init__(self, dataframe, name=None):

        self.column_names = list(dataframe.columns.drop('class'))
        self.column_name_to_id = {name: i for i, name in enumerate(self.column_names)}

        self.features = dataframe.drop('class', axis=1).values
        self.targets = dataframe['class'].values

        self.name = name

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a tuple representing the idx-th sample in the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.LongTensor]: A tuple containing input features and target value.
        """
        return torch.Tensor(self.features[idx]), int(self.targets[idx])
