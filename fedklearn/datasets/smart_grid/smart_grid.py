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

from fedklearn.datasets.smart_grid.constants import *
import torch
import pandas as pd
from torch.utils.data import Dataset

import numpy as np


class FederatedSmartGridDataset:
    """A class representing a federated dataset derived from the SmartGrid dataset.

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

        use_linear (bool): flag to indicate if the data should be processed for a linear model. Default is False.

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

        metadata_path (str): The file path to store the metadata.

        task_id_to_name (dict): A dictionary mapping task IDs to task names.

        use_linear (bool): flag to indicate if the data should be processed for a linear model.



    Methods:
        _init__(self, cache_dir="./", test_frac=None, drop_nationality=True, rng=None):
            Class constructor to initialize the object.

        _download_data(self):
            Downloads the .csv files from the remote server.

        _scale_features(self, df, scaler, mode="train"):
            Scales the features in the DataFrame.

        _scale_target(self, df)
            Scales the target variable in the DataFrame
        _preprocess(self):
            Preprocesses the raw data and saves the intermediate data to the intermediate data folder.

        _generate_tasks_mapping(self):
            Splits the data into tasks and saves the data to the tasks folder.

        _split_data_into_tasks(self, df):
            Splits the SmartGrid dataset across multiple clients based on a specified criterion.

        _iid_divide(self, df):
            Splits a dataframe into a dictionary of dataframes in an iid fashion.

        _save_metadata(self):
            Saves the metadata in a JSON file.

        _load_task_mapping(self):
            Loads the task mapping from a JSON file.

        get_task_dataset(self, task_id, mode="train"):
            Returns an instance of the `SmartGridDataset` class for a specific task and data split type.

        _bmi_divide(self, df):
            Split a dataframe into a dictionary of dataframes based on the BMI feature.

        _correlation_divide(self, df):
            Split a dataframe into a dictionary of dataframes based on the correlation between 'smoker_yes' and 'charges'.


    Examples:
        >>> dataset = FederatedSmartGridDataset(cache_dir="./data", download=True,
        >>>                                       force_generation=True, test_frac=0.1)
        >>> client_train_dataset = federated_data.get_task_dataset(task_id=0, mode="train")
        >>> client_test_dataset = federated_data.get_task_dataset(task_id=0, mode="test")
    """

    def __init__(self, cache_dir="./", download=False, rng=None, force_generation=True, n_tasks=4, split_criterion="random",
                 test_frac=None, scaler="standard", scale_target=False, use_linear=False):
        self.cache_dir = cache_dir
        self.download = download
        self.force_generation = force_generation
        self.n_tasks = n_tasks
        self.split_criterion = split_criterion
        self.scale_target = scale_target

        if use_linear:
            self.raw_data_dir = os.path.join(self.cache_dir, "raw")
            self.intermediate_data_dir = os.path.join(self.cache_dir, "linear", "intermediate")
            self.tasks_folder = os.path.join(self.cache_dir, "linear", "tasks")
        else:
            self.raw_data_dir = os.path.join(self.cache_dir, "raw")
            self.intermediate_data_dir = os.path.join(self.cache_dir, "intermediate")
            self.tasks_folder = os.path.join(self.cache_dir, "tasks")
        self.use_linear = use_linear

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        self.metadata_path = os.path.join(self.tasks_folder, self.split_criterion,  f'{self.n_tasks}',
                                                  "metadata.json")
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

            # print("当前目录:", os.getcwd())
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

        zip_file_path, _ = urllib.request.urlretrieve(ZIP_URL, os.path.join(self.raw_data_dir, "smart_grid.zip"))

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_data_dir)

        df_path = os.path.join(self.raw_data_dir, FILE_NAME)
        df = pd.read_csv(df_path, sep=r'\s*,\s*', engine='python', na_values="?")
        # has_nan = df.isna().any().any()  # 或 df.isnull().values.any()
        # print(f"数据中是否存在 NaN: {has_nan}")

        df.to_csv(os.path.join(self.raw_data_dir, "smart_grid.csv"), index=False)

        # shutil.rmtree(os.path.join(self.raw_data_dir, FOLDER_NAME))
        os.remove(zip_file_path)


    def _scale_features(self, df, mode="train"):
        """
        Scales the features in the DataFrame. If `self.use_linear`
        is True, the data is processed for a linear model.
        Args:
            df (pd.DataFrame): The input DataFrame containing features and target columns.
            mode(str): The mode to use for scaling the features. Default is 'train'.

        Returns:
            pd.DataFrame: The DataFrame containing the scaled features and target columns.

        """
        if self.use_linear:
            categorical_columns = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4', 'stab', 'stabf']
            numerical_columns = [c for c  in df.columns if c not in categorical_columns]

            features_numerical = df[numerical_columns]
            features_categorical = df[categorical_columns]

            stab_column = df["stab"]
            stabf_column = df["stabf"]
            mapping = {"stable": 1, "unstable": 0}
            stabf_column = stabf_column.map(mapping)
            # features_numerical = features_numerical.drop("stab", axis=1)
            # features_numerical = features_numerical.drop("stabf", axis=1)
            # numerical_columns.remove("stab", "stabf")
            
            features_numerical = features_numerical + 1
            features_numerical = pd.concat([features_numerical, stabf_column], axis=1)
            numerical_columns = features_numerical.columns
            # numerical_log = np.log(features_numerical)
            features_numerical = features_numerical.replace(0, np.nan)  # 将零替换为 NaN
            features_numerical = features_numerical.fillna(features_numerical.mean())  # 用均值填充 NaN
            numerical_log = np.log(features_numerical + 1)  # 平移数据以避免零值
            if (features_numerical <= 0).any().any():
                raise ValueError("features_numerical contains invalid values for log transformation.")
            if mode == "train":
                numerical_scaled = self.scaler.fit_transform(numerical_log)
            else:
                numerical_scaled = self.scaler.transform(numerical_log)

            df_scaled = pd.DataFrame(numerical_scaled, columns=numerical_columns)
            df_scaled = pd.concat([features_categorical, df_scaled], axis=1)
            return df_scaled

        else:
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            numerical_columns = numerical_columns[(numerical_columns != 'stab') & (numerical_columns != 'stabf')]
            stabf_column = df['stabf']
            mapping = {"stable": 1, "unstable": 0}
            stabf_column = stabf_column.map(mapping)
            features_numerical = df[numerical_columns]
            if mode == "train":
                features_numerical_scaled = \
                    pd.DataFrame(self.scaler.fit_transform(features_numerical), columns=numerical_columns)
            else:
                features_numerical_scaled = \
                    pd.DataFrame(self.scaler.transform(features_numerical), columns=numerical_columns)

            # Resetting index of both charges_column and features_numerical_scaled
            stabf_column = stabf_column.reset_index(drop=True)
            features_numerical_scaled = features_numerical_scaled.reset_index(drop=True)

            features_scaled = pd.concat([features_numerical_scaled.round(3), stabf_column], axis=1)

            return features_scaled



    def _preprocess(self):
        """Preprocesses the raw data and saves the intermediate data in the intermediate data folder."""

        df = pd.read_csv(os.path.join(self.raw_data_dir, "smart_grid.csv"))
        # print(df.isnull().sum())

        # df = df.dropna(axis=0)
        # for col in df.select_dtypes(include=['object']).columns:
        #     df[col] = df[col].astype('category')
        
        # df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64, sparse=True) # one-hot encoding

        if self.test_frac is None:
            raise ValueError("Please specify the test fraction.")

        train_df, test_df = train_test_split(df, test_size=self.test_frac,
                                             random_state=self.rng.integers(low=0, high=1000))

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df = self._scale_features(train_df, mode="train")
        test_df = self._scale_features(test_df, mode="test")

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
        # mapping = {"stable": 1, "unstable": 0}
        # train_df['stabf'] = train_df['stabf'].map(mapping)
        # test_df['stabf'] = test_df['stabf'].map(mapping)

        # charges_col = pd.concat([train_df['stabf'], test_df['stabf']], axis=0)

        # self.mean_charges = charges_col.mean()
        # self.std_charges = charges_col.std()

        train_tasks_dict = self._split_data_into_tasks(train_df)
        test_tasks_dict = self._split_data_into_tasks(test_df)
        task_dicts = [train_tasks_dict, test_tasks_dict]

        self.task_id_to_name = {f"{i}": task_name for i, task_name in enumerate(train_tasks_dict.keys())}

        for mode, task_dict in zip(["train", "test"], task_dicts):
            for task_name, task_df in task_dict.items():
                task_cache_dir = os.path.join(self.tasks_folder, self.split_criterion, f'{self.n_tasks}', task_name)
                os.makedirs(task_cache_dir, exist_ok=True)

                file_path = os.path.join(task_cache_dir, f"{mode}.csv")
                task_df.to_csv(file_path, index=False)

                logging.debug(f"{mode.capitalize()} data for task '{task_name}' cached at: {file_path}")

        self._save_metadata()



    def _split_data_into_tasks(self, df):
        """
        Splits the SmartGrid dataset across multiple clients based on a specified criterion.
        The available criteria are 'random'
        Args:
            df (pd.DataFrame):  Input DataFrame containing data to split.

        Returns:
            - dict: A dictionary where keys are task names and values are DataFrames for each task.

        """

        split_criterion_dict = {
            "random": self._iid_divide,
            # "correlation": self._correlation_divide,
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

    # def _correlation_divide(self, df):
    #     """
    #     Split a dataframe into a dictionary of dataframes based on the correlation between 'smoker_yes' and 'charges'.
    #     Args:
    #         df(pd.DataFrame): DataFrame to split into tasks.

    #     Returns:
    #         tasks_dict(Dict[str, pd.DataFrame]): A dictionary mapping task IDs to dataframes.
    #     """
    #     task_dict = dict()

    #     no_smoker = min(df['smoker_yes'])
    #     smoker = max(df['smoker_yes'])

    #     if self.n_tasks == 2:
    #         mean_charges = df[(df['smoker_yes'] == smoker)]['charges'].mean()

    #         task_dict['0'] = df[(df['smoker_yes'] == no_smoker) & (df['charges'] <= mean_charges)]
    #         task_dict['0'] = pd.concat([task_dict['0'],
    #                                     df[(df['smoker_yes'] == smoker) & (df['charges'] > mean_charges)]])

    #         task_dict['1'] = df[(df['smoker_yes'] == no_smoker) & (df['charges'] > mean_charges)]
    #         task_dict['1'] = pd.concat([task_dict['1'],
    #                                     df[(df['smoker_yes'] == smoker) & (df['charges'] <= mean_charges)]])
    #     else:
    #         raise ValueError("Correlation-based split is only supported for 2 tasks.")

    #     return task_dict


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

    def _save_metadata(self):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "w") as f:
            metadata_dict = {'split_criterion': self.split_criterion,
                              'n_tasks': self.n_tasks,
                              'cache_dir': os.path.abspath(self.cache_dir),
                              'task_mapping': self.task_id_to_name, 
                            #   'mean': self.mean_charges,
                            #   'std': self.std_charges
                              }
            json.dump(metadata_dict, f)


    def _load_task_mapping(self):
        """
        Load the task mapping from a JSON file.
        """
        with open(self.metadata_path, "r") as f:
            metadata_dict = json.load(f)
            self.task_id_to_name = metadata_dict['task_mapping']


    def get_task_dataset(self, task_id, mode="train"):
        """
        Returns an instance of the `SmartGridDataset` class for a specific task and data split type.

        Args:
            task_id (int or str): The task number.
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            SmartGridDataset: An instance of the `SmartGridDataset` class representing the specified task and data split.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_id = str(task_id)

        task_name = self.task_id_to_name[task_id]
        task_cache_dir = os.path.join(self.tasks_folder, self.split_criterion,f'{self.n_tasks}', task_name)
        file_path = os.path.join(task_cache_dir, f"{mode}.csv")
        task_data = pd.read_csv(file_path)

        if self.scale_target:
            task_data = self._scale_target(task_data)
        return SmartGridDataset(task_data, name=task_name)
    
    def _scale_target(self, df):
        """
        Scale the target variable.
        Args:
            df(pd.DataFrame): DataFrame to scale.
        Returns:
            df(pd.DataFrame): Scaled DataFrame.
        """
        with open(self.metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # self.mean_charges = metadata_dict["mean"]
        # self.std_charges = metadata_dict["std"]
        # df['stabf'] = (df['stabf'] - self.mean_charges) / self.std_charges
        return df


class SmartGridDataset(Dataset):
    """
    PyTorch Dataset class for the SmartGrid dataset.

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

    def __init__(self, dataframe, name="smart_grid"):

        self.column_names = list(dataframe.columns.drop("stabf"))
        self.column_name_to_id = {name: i for i, name in enumerate(self.column_names)}

        # self.features = dataframe.drop(["stab"], axis=1).values
        self.features = dataframe.drop(["stabf"], axis=1).values
        # status_map = {"stable": 1, "unstable": 0}
        self.targets = dataframe["stabf"].values

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
    dataset = FederatedSmartGridDataset(cache_dir="../../../scripts/data/smart_grid", download=True,
                                          force_generation=True, test_frac=0.1)
    print("Data loaded successfully.")
