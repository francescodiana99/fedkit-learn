import os
import ssl
import urllib
import logging

import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset

from .constants import *


class FederatedAdultDataset:
    """
    A class representing a federated dataset derived from the Adult dataset.

    This dataset is designed for federated learning scenarios where the data is split across multiple clients,
    and each client represents a specific task based on criteria such as age and education level.

    Args:
        cache_dir (str, optional): The directory path for caching downloaded and preprocessed data. Default is "./".

        test_frac (float, optional): Fraction of the test samples; it should be a float between 0 and 1.
            If `None`, the original train-test split is applied. Default is `None`.

        drop_nationality (bool, optional): Flag to drop the nationality column from the data. Default is `True`.

        scaler_name (str, optional): Name of the scaler used to scale numerical features.
            Default is "standard". It can be "min_max" or "standard".

        rng (Random Number Generator, optional): An instance of a random number generator.
            If `None`, a new generator will be created. Default is `None`.

        download (bool, optional): Whether to download the data if not already cached. Default is True.


    Attributes:
        cache_dir (str): The directory path for caching downloaded and preprocessed data.

        test_frac (float): Fraction of the test samples; it should be a float between 0 and 1.
            If `None`, the original train-test split is applied.

        drop_nationality (bool): Flag to drop the nationality column from the data. Default is `True`.

        scaler (StandardScaler): An instance of `StandardScaler` used to scale numerical features.

        task_id_to_name (dict): A mapping of task IDs to task names.

        rng (Random Number Generator): An instance of a random number generator.
            If `None`, a new generator will be created.

    Methods:
        __init__(self, cache_dir="./", test_frac=None, drop_nationality=True, rng=None):
            Class constructor to initialize the object.

        _transform_education_level(x):
            A static method to transform the education level.

        _scale_features(self, df, scaler, mode="train"):
            Scale numerical features of the DataFrame.
            df: Input DataFrame.
            scaler: The scaler object.
            mode: Either "train" or "test" to determine if fitting or transforming.

        _download_and_preprocess(self):
            Download the Adult dataset and preprocess it.
            Returns scaled features of the training and testing datasets.

        _split_data_into_tasks(self, df):
            Split the Adult dataset across multiple clients based on specified criteria.
            Returns a dictionary where keys are task names or numbers, and values are DataFrames for each task.

        get_task_dataset(self, task_number, mode='train'):
            Returns an instance of the `AdultDataset` class for a specific task and data split type.
            task_number: The task number or name.
            mode: The type of data split, either 'train' or 'test'. Default is 'train'.


    Examples:
        >>> federated_data = FederatedAdultDataset(cache_dir="./data", test_frac=0.2)
        >>> client_train_dataset = federated_data.get_task_dataset(task_id=0, mode="train")
        >>> client_test_dataset = federated_data.get_task_dataset(task_id=0, mode="test")
    """
    def __init__(
            self, cache_dir="./", test_frac=None, drop_nationality=True, scaler_name="standard", download=True, rng=None
    ):
        """
        Raises:
            FileNotFoundError: If processed data folders are not found and download is set to False.
        """

        self.cache_dir = cache_dir
        self.test_frac = test_frac
        self.drop_nationality = drop_nationality
        self.download = download
        self.scaler_name = scaler_name

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        tasks_folder = os.path.join(self.cache_dir, 'tasks')

        if os.path.exists(tasks_folder):
            logging.info("Processed data folders found in the tasks directory. Loading existing files.")
            self._load_task_mapping()

        elif not self.download:
            raise FileNotFoundError(
                "Processed data folders not found. Set 'download' to True to download the data."
            )

        else:
            self.scaler = self.set_scaler(self.scaler_name)

            train_df, test_df = self._download_and_preprocess()

            train_tasks_dict = self._split_data_into_tasks(train_df)
            test_tasks_dict = self._split_data_into_tasks(test_df)

            task_dicts = [train_tasks_dict, test_tasks_dict]

            self.task_id_to_name = {i: task_name for i, task_name in enumerate(train_tasks_dict.keys())}

            for mode, task_dict in zip(['train', 'test'], task_dicts):
                for task_name, task_data in task_dict.items():
                    task_cache_dir = os.path.join(self.cache_dir, 'tasks', task_name)
                    os.makedirs(task_cache_dir, exist_ok=True)

                    file_path = os.path.join(task_cache_dir, f'{mode}.csv')
                    task_data.to_csv(file_path, index=False)

                    logging.debug(f"{mode.capitalize()} data for task '{task_name}' cached at: {file_path}")

    def _load_task_mapping(self):
        task_dir = os.path.join(self.cache_dir, 'tasks')
        task_names = [
            dir_name for dir_name in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, dir_name))
        ]

        self.task_id_to_name = {i: task_name for i, task_name in enumerate(task_names)}

    @staticmethod
    def set_scaler(scaler_name):
        if scaler_name == "min_max":
            return MinMaxScaler()
        elif scaler_name == "standard":
            return StandardScaler()
        else:
            raise NotImplementedError(f"Scaler {scaler_name} is not implemented.")

    @staticmethod
    def _transform_education_level(x):
        if x == "HS-grad":
            return "HS-grad"
        elif (x == "Bachelors") or (x == "Some-college"):
            return "Bachelors"
        elif x in {'10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Preschool'}:
            return "Compulsory"
        elif (x == "Assoc-acdm") or (x == "Assoc-voc"):
            return "Associate"
        else:
            return x

    @staticmethod
    def _scale_features(df, scaler, mode="train"):
        numerical_columns = df.select_dtypes(include=['number']).columns

        numerical_columns = numerical_columns[numerical_columns != 'income']

        income_col = df['income']
        education_col = df['education']
        age_col = df['age']

        features_numerical = df[numerical_columns]

        if mode == "train":
            features_numerical_scaled = \
                pd.DataFrame(scaler.fit_transform(features_numerical), columns=numerical_columns)
        else:
            features_numerical_scaled = \
                pd.DataFrame(scaler.transform(features_numerical), columns=numerical_columns)

        features_numerical_scaled['age_scaled'] = features_numerical_scaled['age']

        features_numerical_scaled = features_numerical_scaled.drop('age', axis=1)

        features_scaled = pd.concat([education_col, age_col, features_numerical_scaled, income_col], axis=1)

        return features_scaled

    @staticmethod
    def _split_data_into_tasks(df):
        """ Split the adult dataset across multiple clients based on specified criteria.

        The input should have columns "age" and "education", with possible values:
            {"Doctorate," "Prof-school", "Masters", "Bachelors", "Associate", "HS-grad",  "Compulsory"}

        Args:
         - df (pd.DataFrame):  Input DataFrame containing the columns "age" and "education".

        Raises:
            ValueError: If the input DataFrame does not contain the required columns ('age' and 'education').

        Returns:
            - dict: A dictionary where keys are task names and values are DataFrames for each task.
        """
        required_columns = {'age', 'education'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input DataFrame must contain columns {', '.join(required_columns)}.")

        tasks_dict = dict()
        for task_name, criteria in SPLIT_CRITERIA.items():
            try:
                task_indices = df.index[
                    (df['age'].between(*criteria['age'])) & (df['education'] == criteria['education'])
                ].tolist()

                task_df = df.loc[task_indices]
                task_df = task_df.drop(['education', 'age'], axis=1)

                tasks_dict[task_name] = task_df

            except KeyError:
                raise ValueError(
                    f"Invalid criteria structure for task '{task_name}'."
                    f" Ensure 'age' and 'education' are specified in the criteria."
                )

        return tasks_dict

    def _download_and_preprocess(self):
        """ Download the adult dataset and preprocess it.

        The pre-processing involves the following steps:
            * Drop the 'fnlwgt' column
            * Drop the nationality column if `drop_nationality` is True
            * Drop columns with missing data
            * Replace 'income' column with binary values
            * Transform 'education' column to have 7 values:
                {"Doctorate," "Prof-school", "Masters", "Bachelors", "Associate", "HS-grad",  "Compulsory"}
            * Get dummy variables for categorical features
            * Train/test split
            * Scale the data

        Remark: We keep the original 'age' and the transformed 'education' columns because they are needed to split
        data across clients in a later stage.

        Args:
            - test_frac (float, optional): Fraction of the test samples; it should be a float between 0 and 1.
                If `None`, the original train-test split is applied. Default is `None`.
            - drop_nationality (bool, optional): Flag to drop the nationality column from the data. Default is `True`.
            - rng (Random Number Generator):
            - cache_dir (str, optional): directory to cache the downloaded file

        Returns:
            - pd.DataFrame: Scaled features of the training dataset, including the transformed 'education'
                column and scaled numerical features. The 'income' column is preserved as the original target variable.

            - pd.DataFrame: Scaled features of the testing dataset, including the transformed 'education'
                column and scaled numerical features. The 'income' column is preserved as the original target variable.

        """
        try:
            train_df = pd.read_csv(TRAIN_URL, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
            test_df = pd.read_csv(TEST_URL, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?", skiprows=1)

        except urllib.error.URLError:

            ssl._create_default_https_context = ssl._create_unverified_context

            import zipfile

            os.makedirs(os.path.join(self.cache_dir, 'raw'), exist_ok=True)

            zip_file_path, _ = urllib.request.urlretrieve(BACKUP_URL, os.path.join(self.cache_dir, 'raw', "adult.zip"))

            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.cache_dir, 'raw'))

            train_path = os.path.join(self.cache_dir, 'raw', "adult.data")
            test_path = os.path.join(self.cache_dir, 'raw', "adult.test")

            train_df = pd.read_csv(train_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
            test_df = pd.read_csv(test_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?", skiprows=1)

        if self.cache_dir is not None:
            os.makedirs(os.path.join(self.cache_dir, 'raw'), exist_ok=True)

            raw_train_path = os.path.join(self.cache_dir, 'raw', 'train.csv')
            train_df.to_csv(raw_train_path, index=False)
            logging.debug(f"Raw train data cached at: {raw_train_path}")

            raw_test_path = os.path.join(self.cache_dir, 'raw', 'test.csv')
            test_df.to_csv(raw_test_path, index=False)
            logging.debug(f"Raw test data cached at: {raw_test_path}")

        num_train = len(train_df)

        df = pd.concat([train_df, test_df])

        df = df.drop('fnlwgt', axis=1)  # irrelevant fo the prediction task

        if self.drop_nationality:
            df = df.drop('native-country', axis=1)
            CATEGORICAL_COLUMNS.remove('native-country')

        df['income'] = df['income'].replace('<=50K', 0).replace('>50K', 1)
        df['income'] = df['income'].replace('<=50K.', 0).replace('>50K.', 1)

        df["education"] = df["education"].apply(self._transform_education_level)

        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        if self.test_frac is None:
            train_df, test_df = df[:num_train], df[num_train:]
        else:
            train_df, test_df = train_test_split(df, test_size=self.test_frac, random_state=self.rng)

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df = self._scale_features(train_df, self.scaler, mode="train")
        test_df = self._scale_features(test_df, self.scaler, mode="test")

        if self.cache_dir is not None:
            os.makedirs(os.path.join(self.cache_dir, 'intermediate'), exist_ok=True)
            processed_train_path = os.path.join(self.cache_dir, 'intermediate', 'train.csv')
            processed_test_path = os.path.join(self.cache_dir, 'intermediate', 'test.csv')
            train_df.to_csv(processed_train_path, index=False)
            test_df.to_csv(processed_test_path, index=False)
            logging.debug(f"Processed train data cached at: {processed_train_path}")
            logging.debug(f"Processed test data cached at: {processed_test_path}")

        return train_df, test_df

    def get_task_dataset(self, task_id, mode='train'):
        """
        Returns an instance of the `AdultDataset` class for a specific task and data split type.

        Args:
            task_id (int): The task number.
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            AdultDataset: An instance of the `AdultDataset` class representing the specified task and data split.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_name = self.task_id_to_name[task_id]
        task_cache_dir = os.path.join(self.cache_dir, 'tasks', task_name)
        file_path = os.path.join(task_cache_dir, f'{mode}.csv')
        task_data = pd.read_csv(file_path)

        return AdultDataset(task_data, name=task_name)

    def get_pooled_data(self, mode="train"):
        """
        Returns the pooled dataset before splitting into tasks.

        Args:
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            AdultDataset: An instance of the `AdultDataset` class containing the pooled data.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        file_path = os.path.join(self.cache_dir, 'intermediate', f'{mode}.csv')

        data = pd.read_csv(file_path)

        data = data.drop(['education', 'age'], axis=1)

        return AdultDataset(data, name="pooled")


class AdultDataset(Dataset):
    """
     PyTorch Dataset class for the Adult dataset.

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
    def __init__(self, dataframe, name=None):
        """
        Initializes the AdultDataset.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing features and targets.
            name (str or None): Name or identifier of the dataset. Default is `None`.
        """
        self.column_names = list(dataframe.columns.drop('income'))
        self.column_name_to_id = {name: i for i, name in enumerate(self.column_names)}

        self.features = dataframe.drop('income', axis=1).values
        self.targets = dataframe['income'].values

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
