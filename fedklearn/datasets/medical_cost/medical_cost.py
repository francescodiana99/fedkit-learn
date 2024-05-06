import json
import os
import shutil
import logging
import urllib
import zipfile

import requests
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fedklearn.datasets.medical_cost.constants import *
import torch
import pandas as pd
from torch.utils.data import Dataset

import numpy as np


class FederatedMedicalCostDataset:
    """A class representing a federated dataset derived from the MedicalCost dataset.

    This dataset is designed for federated learning scenarios where the data is split across multiple clients,
    and each client represents a specific task.

    Args:
        cache_dir (str): The directory path to store the downloaded and processed data. Default is './'.

        download (bool): Whether to download the data. Default is True.

        rng (np.random.Generator): A random number generator. Default is None.

        force_generation (bool): Forces the generation of tasks. Default is True.

        n_tasks (int): The number of tasks to split the data into. Default is 4.

        split_criterion (str): The criterion to use for splitting the data into tasks. Default is 'random'.

        test_frac (float): The fraction of data to use for testing. Default is None.

        scaler (str): The type of scaler to use for scaling the features. Default is 'standard'.

        scale_target (bool): flag to indicate if the target column should be scaled. Default is True.

    Attributes:
        cache_dir (str): The directory path to store the downloaded and processed data.

        download (bool): Whether to download the data.

        force_generation (bool): Forces the generation of tasks.

        n_tasks (int): The number of tasks to split the data into.

        split_criterion (str): The criterion to use for splitting the data into tasks.

        test_frac (float): The fraction of data to use for testing.

        scaler (str): The type of scaler to use for scaling the features.

        scale_target (bool): flag to indicate if the target column should be scaled.

        raw_data_dir (str): The directory path to store the raw data.

        intermediate_data_dir (str): The directory path to store the intermediate data.

        tasks_folder (str): The directory path to store the tasks.

        rng (np.random.Generator): A random number generator.

        _split_criterion_path (str): The file path to store the split criterion.

        _metadata_path (str): The file path to store the metadata.

    Methods:
        _init__(self, cache_dir="./", test_frac=None, drop_nationality=True, rng=None):
            Class constructor to initialize the object.

        _download_data(self):
            Downloads the .csv files from the remote server.

        _scale_features(self, df, scaler, mode="train"):
            Scales the features in the DataFrame.

        _preprocess(self):
            Preprocesses the raw data and saves the intermediate data to the intermediate data folder.

        _generate_tasks_mapping(self):
            Splits the data into tasks and saves the data to the tasks folder.

        _split_data_into_tasks(self, df):
            Splits the MedicalCost dataset across multiple clients based on a specified criterion.

        _iid_divide(self, df):
            Splits a dataframe into a dictionary of dataframes in an iid fashion.

        _save_split_criterion(self):
            Saves the split criterion in a JSON file.

        _save_task_mapping(self, metadata_dict):
            Saves the task mapping in a JSON file.

        _load_task_mapping(self):
            Loads the task mapping from a JSON file.

        get_task_dataset(self, task_id, mode="train"):
            Returns an instance of the `MedicalCostDataset` class for a specific task and data split type.

    Examples:
        >>> dataset = FederatedMedicalCostDataset(cache_dir="./data", download=True,
        >>>                                       force_generation=True, test_frac=0.1)
        >>> client_train_dataset = federated_data.get_task_dataset(task_id=0, mode="train")
        >>> client_test_dataset = federated_data.get_task_dataset(task_id=0, mode="test")
    """

    def __init__(self, cache_dir="./", download=True, rng=None, force_generation=True, n_tasks=4, split_criterion="random",
                 test_frac=None, scaler="standard", scale_target=True):
        self.cache_dir = cache_dir
        self.download = download
        self.force_generation = force_generation
        self.n_tasks = n_tasks
        self.split_criterion = split_criterion
        self.scale_target = scale_target

        self.raw_data_dir = os.path.join(self.cache_dir, "raw")
        self.intermediate_data_dir = os.path.join(self.cache_dir, "intermediate")
        self.tasks_folder = os.path.join(self.cache_dir, "tasks")

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        self._split_criterion_path = os.path.join(self.cache_dir, "split_criterion.json")
        self._metadata_path = os.path.join(self.cache_dir, "metadata.json")
        self.test_frac = test_frac

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaler: {scaler}. Available scalers are 'standard' and 'minmax'.")

        if os.path.exists(self.tasks_folder) and not self.force_generation:
            logging.info("Processed data folders found. Loading existing files..")
            self._load_task_mapping()

        elif not self.download and self.force_generation:
            logging.info("Data found in the cache directory. Splitting data into tasks..")
            self._generate_tasks_mapping()

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
            self._preprocess()
            self._generate_tasks_mapping()

    def _download_data(self):
        """Downloads the .csv files from the remote server."""

        os.makedirs(self.raw_data_dir, exist_ok=True)

        zip_file_path, _ = urllib.request.urlretrieve(ZIP_URL, os.path.join(self.raw_data_dir, "medical_cost.zip"))

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_data_dir)

        df_path = os.path.join(self.raw_data_dir,
                               "82a9f1c1c473d6585e750ad2e3c05a41-d42d226d0dd64e7f5395a0eec1b9190a10edbc03",
                               "Medical_Cost.csv")
        df = pd.read_csv(df_path, sep=r'\s*,\s*', engine='python', na_values="?")

        df.to_csv(os.path.join(self.raw_data_dir, "medical_cost.csv"), index=False)

        shutil.rmtree(os.path.join(self.raw_data_dir,
                               "82a9f1c1c473d6585e750ad2e3c05a41-d42d226d0dd64e7f5395a0eec1b9190a10edbc03"))
        os.remove(zip_file_path)


    @staticmethod
    def _scale_features( df, scaler, mode="train", scale_target=True):
        """
        Scales the features in the DataFrame. If `scale_target` is True, the target column is also scaled.
        Args:
            df (pd.DataFrame): The input DataFrame containing features and target columns.
            scaler (sklearn.preprocessing.object): The scaler object to use for scaling the features.
            Default is StandardScaler.
            mode(str): The mode to use for scaling the features. Default is 'train'.
            scale_target(bool): flag to indicate if the target column should be scaled. Default is True.

        Returns:
            pd.DataFrame: The DataFrame containing the scaled features and target columns.

        """
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = numerical_columns[numerical_columns != 'charges']
        charges_column = df['charges']
        features_numerical = df[numerical_columns]

        if not scale_target:
            if mode == "train":
                features_numerical_scaled = \
                    pd.DataFrame(scaler.fit_transform(features_numerical), columns=numerical_columns)
            else:
                features_numerical_scaled = \
                    pd.DataFrame(scaler.transform(features_numerical), columns=numerical_columns)

            # Resetting index of both charges_column and features_numerical_scaled
            charges_column = charges_column.reset_index(drop=True)
            features_numerical_scaled = features_numerical_scaled.reset_index(drop=True)

            features_scaled = pd.concat([features_numerical_scaled, charges_column], axis=1)

            return features_scaled

        else:
            if mode =="train":
                df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            else:
                df = pd.DataFrame(scaler.transform(df), columns=df.columns)
            return df


    def _preprocess(self):
        """Preprocesses the raw data and saves the intermediate data to the intermediate data folder."""

        df = pd.read_csv(os.path.join(self.raw_data_dir, "medical_cost.csv"))

        df = df.dropna(axis=0)
        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        if self.test_frac is None:
            raise ValueError("Please specify the test fraction.")

        train_df, test_df = train_test_split(df, test_size=self.test_frac,
                                             random_state=self.rng.integers(low=0, high=1000))

        train_df = self._scale_features(train_df, self.scaler, mode="train", scale_target=self.scale_target)
        test_df = self._scale_features(test_df, self.scaler, mode="test", scale_target=self.scale_target)

        if self.intermediate_data_dir is not None:
            os.makedirs(self.intermediate_data_dir, exist_ok=True)

            train_df.to_csv(os.path.join(self.intermediate_data_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(self.intermediate_data_dir, "test.csv"), index=False)

            logging.debug(f"Processed data cached in: {self.intermediate_data_dir}")


    def _generate_tasks_mapping(self):
        """
        Splits the data into tasks and saves the data to the tasks folder.
        """
        train_df = pd.read_csv(os.path.join(self.intermediate_data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.intermediate_data_dir, "test.csv"))

        train_tasks_dict = self._split_data_into_tasks(train_df)
        test_tasks_dict = self._split_data_into_tasks(test_df)
        task_dicts = [train_tasks_dict, test_tasks_dict]

        self.task_id_to_name = {f"{i}": task_name for i, task_name in enumerate(train_tasks_dict.keys())}

        for mode, task_dict in zip(["train", "test"], task_dicts):
            for task_name, task_df in task_dict.items():
                task_cache_dir = os.path.join(self.tasks_folder, self.split_criterion, task_name)
                os.makedirs(task_cache_dir, exist_ok=True)

                file_path = os.path.join(task_cache_dir, f"{mode}.csv")
                task_df.to_csv(file_path, index=False)

                logging.debug(f"{mode.capitalize()} data for task '{task_name}' cached at: {file_path}")

        self._save_task_mapping(self.task_id_to_name)

        self._save_split_criterion()


    def _split_data_into_tasks(self, df):
        """
        Splits the MedicalCost dataset across multiple clients based on a specified criterion.
        The available criteria are 'random'
        Args:
            df (pd.DataFrame):  Input DataFrame containing data to split.

        Returns:
            - dict: A dictionary where keys are task names and values are DataFrames for each task.

        """

        split_criterion_dict = {
            "random": self._iid_divide,
            "correlation": self._correlation_divide,
            "bmi": self._bmi_divide,
        }
        if self.split_criterion in split_criterion_dict:
            tasks_dict = split_criterion_dict[self.split_criterion](df)
        else:
            raise ValueError(f"Invalid split criterion: {self.split_criterion}. Available criteria are "
                             f"'random', 'correlation', 'bmi'.")
        return tasks_dict


    def _bmi_divide(self, df):
        """
        Split a dataframe into a dictionary of dataframes based on the BMI feature.
        Args:
            df(pd.DataFrame): DataFrame to split into tasks.

        Returns:
            tasks_dict(Dict[str, pd.DataFrame]): A dictionary mapping task IDs to dataframes.
        """
        task_dict = dict()
        num_elems = len(df)
        min_bmi = min(df['bmi'])
        max_bmi = max(df['bmi'])

        interval_size = (max_bmi - min_bmi) / self.n_tasks
        i = min_bmi
        j = i + interval_size

        for task_id in range(self.n_tasks):
            task_dict[f"{task_id}"] = df[(df['bmi'] >= i) & (df['bmi'] < j)]
            i = j
            j = i + interval_size
        return task_dict

    def _correlation_divide(self, df):
        """
        Split a dataframe into a dictionary of dataframes based on the correlation between 'smoker_yes' and 'charges'.
        Args:
            df(pd.DataFrame): DataFrame to split into tasks.

        Returns:
            tasks_dict(Dict[str, pd.DataFrame]): A dictionary mapping task IDs to dataframes.
        """
        task_dict = dict()

        no_smoker = min(df['smoker_yes'])
        smoker = max(df['smoker_yes'])

        if self.n_tasks == 2:
            mean_charges = df[(df['smoker_yes'] == smoker)]['charges'].mean()

            task_dict['0'] = df[(df['smoker_yes'] == no_smoker) & (df['charges'] <= mean_charges)]
            task_dict['0'] = pd.concat([task_dict['0'],
                                        df[(df['smoker_yes'] == smoker) & (df['charges'] > mean_charges)]])

            task_dict['1'] = df[(df['smoker_yes'] == no_smoker) & (df['charges'] > mean_charges)]
            task_dict['1'] = pd.concat([task_dict['1'],
                                        df[(df['smoker_yes'] == smoker) & (df['charges'] <= mean_charges)]])
        else:
            raise ValueError("Correlation-based split is only supported for 2 tasks.")

        return task_dict


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

    def _save_split_criterion(self):
        with open(self._split_criterion_path, "w") as f:
            criterion_dict = {'split_criterion': self.split_criterion, 'n_tasks': self.n_tasks}
            json.dump(criterion_dict, f)


    def _save_task_mapping(self, metadata_dict):
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
        with (open(self._metadata_path, "r") as f):
            metadata = json.load(f)
            self.task_id_to_name = metadata[self.split_criterion]


    def get_task_dataset(self, task_id, mode="train"):
        """
        Returns an instance of the `MedicalCostDataset` class for a specific task and data split type.

        Args:
            task_id (int or str): The task number.
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            MedicalCostDataset: An instance of the `MedicalCostDataset` class representing the specified task and data split.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_id = str(task_id)

        task_name = self.task_id_to_name[task_id]
        task_cache_dir = os.path.join(self.tasks_folder, self.split_criterion, task_name)
        file_path = os.path.join(task_cache_dir, f"{mode}.csv")
        task_data = pd.read_csv(file_path)
        return MedicalCostDataset(task_data, name=task_name)


class MedicalCostDataset(Dataset):
    """
    PyTorch Dataset class for the MedicalCost dataset.

    Args:
         dataframe (pd.DataFrame): The input DataFrame containing features and targets.
         name (str, optional): A string representing the name or identifier of the dataset. Default is `None`.

    Attributes:
         features (numpy.ndarray): Array of input features excluding the 'income' column.
         targets (numpy.ndarray): Array of target values from the 'income' column.
         name (str or None): Name or identifier of the dataset. Default is `None`.
         column_names (list): List of original column names excluding the 'income' column.
         column_name_to_id (dict): Dictionary mapping column names to numeric ids.

    Methods:
         __len__(): Returns the number of samples in the dataset.
         __getitem__(idx): Returns a tuple representing the idx-th sample in the dataset.
    """

    def __init__(self, dataframe, name="medical_cost"):

        self.column_names = list(dataframe.columns.drop("charges"))
        self.column_name_to_id = {name: i for i, name in enumerate(self.column_names)}

        self.features = dataframe.drop(["charges"], axis=1).values
        self.targets = dataframe["charges"].values

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
        return torch.Tensor(self.features[idx]), np.float32(self.targets[idx])


if __name__ == "__main__":
    dataset = FederatedMedicalCostDataset(cache_dir="../../../scripts/data/medical_cost", download=True,
                                          force_generation=True, test_frac=0.1)
    print("Data loaded successfully.")
