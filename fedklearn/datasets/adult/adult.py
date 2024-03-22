import os
import shutil
import ssl
import json
import urllib
import logging

import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from  fedklearn.attacks.aia import ModelDrivenAttributeInferenceAttack

from torch.utils.data import Dataset, DataLoader

from torch import nn

from fedklearn.models.linear import LinearLayer

from .constants import *
from ...metrics import binary_accuracy_with_sigmoid, threshold_binary_accuracy
from ...trainer.trainer import Trainer, DebugTrainer


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

        split_criterion (str, optional): The criterion used to split the data into tasks. Default is 'age_education'.

        n_tasks (int, optional): The number of tasks to split the data into. Default is `None`.

        n_task_samples (int, optional): The number of samples per task. Default is `None`.

        force_generation (bool, optional): Whether to force the generation of the dataset. Default is `False`.

        seed (int, optional): The seed for the random number generator. Default is 42.

        binarize_marital_status (bool, optional): Whether to binarize the marital status. Default is `False`.

        binarize_race (bool, optional): Whether to binarize the race. Default is `False`.

        device (str, optional): The device to use for training the model. Default is 'cpu'.

        sensitive_attribute_id (int, optional): The index of the sensitive attribute in the dataset. Default is `None`.


    Attributes:
        cache_dir (str): The directory path for caching downloaded and preprocessed data.

        test_frac (float): Fraction of the test samples; it should be a float between 0 and 1.
            If `None`, the original train-test split is applied.

        drop_nationality (bool): Flag to drop the nationality column from the data. Default is `True`.

        scaler (StandardScaler): An instance of `StandardScaler` used to scale numerical features.

        task_id_to_name (dict): A mapping of task IDs to task names.

        rng (Random Number Generator): An instance of a random number generator.
            If `None`, a new generator will be created.

        _metadata_path (str): JSON file storing the metadata.

        _split_criterion_path (str): JSON file storing the split criterion.

    Methods:
        __init__(self, cache_dir="./", test_frac=None, drop_nationality=True, rng=None):
            Class constructor to initialize the object.

        _transform_education_level(x):
            A static method to transform the education level.
            
        _transform_marital_status(x):
            A static method to transform the marital status.

        _transform_race(x):
            A static method to transform the race.

        _scale_features(self, df, scaler, mode="train"):
            Scale numerical features of the DataFrame.
            df: Input DataFrame.
            scaler: The scaler object.
            mode: Either "train" or "test" to determine if fitting or transforming.

        _download_and_preprocess(self):
            Download the Adult dataset and preprocess it.
            Returns scaled features of the training and testing datasets.

        _save_task_mapping(self, metadata_dict):
            Save the task mapping to a JSON file.

        _load_task_mapping(self):
            Load the task mapping from a JSON file.

        _save_split_criterion(self, split_criterion):
            Save the split criterion to a JSON file.

        _split_by_age_education(self, df):
            Split the data based on age and education level.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_age(self, df):
            Split the data based on age.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_num_tasks(self, df):
            Split the data into a specified number of tasks.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_num_tasks_and_labels(self, df):
            Split the data into a specified number of tasks, maintaining both the labels in the datasets.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_kmeans(self, df):
            Split the data using k-means clustering.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_gmm(self, df):
            Split the data using Gaussian Mixture Model.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_prediction(self, df):
            Split the data according to the prediction of a Linear model.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

        _split_by_aia(self, df):
            Split the data according to the attribute inference attack.
            Returns a dictionary where keys are task names and values are DataFrames for each task.

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
            self, cache_dir="./", test_frac=None, drop_nationality=True, scaler_name="standard", download=True,
            rng=None, split_criterion='age_education', n_tasks=None, n_task_samples=None, force_generation=False,
            seed=42, binarize_marital_status=False, binarize_race=False, device='cpu', sensitive_attribute_id=None
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
        self.split_criterion = split_criterion
        self.n_tasks = n_tasks
        self.n_task_samples = n_task_samples
        self.force_generation = force_generation
        self.seed = seed
        self.binarize_marital_status = binarize_marital_status,
        self.binarize_race = binarize_race
        self.device = device
        self.sensitive_attribute_id = sensitive_attribute_id

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        tasks_folder = os.path.join(self.cache_dir, 'tasks', self.split_criterion)

        self._metadata_path = os.path.join(self.cache_dir, "metadata.json")

        self._split_criterion_path = os.path.join(self.cache_dir, "split_criterion.json")

        # TODO: avoid to force generation
        if os.path.exists(tasks_folder) and not self.force_generation:
            logging.info("Processed data folders found in the tasks directory. Loading existing files.")
            self._load_task_mapping()


        elif not self.download:
            raise FileNotFoundError(
                "Processed data folders not found. Set 'download' to True to download the data."
            )

        else:
            logging.info("Forcing data generation....")
            # remove the task folder if it exists to avoid inconsistencies
            if os.path.exists(tasks_folder):
                shutil.rmtree(tasks_folder)

            self.scaler = self.set_scaler(self.scaler_name)

            train_df, test_df = self._download_and_preprocess()

            if self.split_criterion in ['prediction', 'aia']:

                train_df = train_df.drop(['education', 'age'], axis=1)
                test_df = test_df.drop(['education', 'age'], axis=1)

                train_dataset = AdultDataset(train_df)
                test_dataset = AdultDataset(test_df)

                train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

                self.split_model = self._train_splitting_model(train_loader=train_loader, test_loader=test_loader,
                                                               device=self.device)

            train_tasks_dict = self._split_data_into_tasks(train_df)
            test_tasks_dict = self._split_data_into_tasks(test_df)

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
    def _transform_marital_status(x):
        if x in {'Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent'}:
            return 1
        else:
            return 0


    @staticmethod
    def _transform_race(x):
        if x == "White":
            return 1
        else:
            return 0


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


    def _train_splitting_model(self, train_loader, test_loader, device):
        """Train a model to split the data into tasks."""

        logging.info(f"Training a linear model to split the data...")

        if self.binarize_marital_status:
            linear_model = LinearLayer(input_dimension=36, output_dimension=1)
        else:
            linear_model = LinearLayer(input_dimension=41, output_dimension=1)
        trainer = DebugTrainer(model=linear_model,
                          criterion=torch.nn.BCEWithLogitsLoss(),
                          optimizer=torch.optim.SGD(linear_model.parameters(), lr=0.02),
                          device=device,
                          metric=binary_accuracy_with_sigmoid,
                          is_binary_classification=True
                          )
        trainer.fit_epochs(loader=train_loader, n_epochs=100)
        train_loss, train_metric = trainer.evaluate_loader(train_loader)
        test_loss, test_metric = trainer.evaluate_loader(test_loader)

        logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
        logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")

        return linear_model


    def _get_model_split(self, linear_model, dataloader, columns, device):
        """Split the data based on the prediction of a linear model."""
        tasks_dict = dict()
        tasks_dict["0"] = pd.DataFrame(columns=columns)
        tasks_dict["1"] = pd.DataFrame(columns=columns)

        linear_model.eval()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = torch.round(torch.sigmoid(linear_model(x)))

            y_pred = y_pred.squeeze()

            prediction_mask = torch.eq(y_pred, y).type(torch.float32)
            tasks_data = torch.cat((x,y.unsqueeze(1), prediction_mask.unsqueeze(1)), dim=1).cpu().numpy()
            tasks_df = pd.DataFrame(tasks_data)

            tasks_df_0 = tasks_df[columns[-1] == 0].drop(columns[-1], axis=1)
            tasks_df_0.columns = columns

            tasks_df_1 = tasks_df[columns[-1] == 0].drop(columns[-1], axis=1)
            tasks_df_1.columns = columns

            # TODO: it is slow, check if there is a way to speed up
            tasks_dict["0"] = pd.concat([tasks_dict["0"], tasks_df_0], axis=0)
            tasks_dict["1"] = pd.concat([tasks_dict["1"], tasks_df_1], axis=0)

        return tasks_dict


    def _split_by_age_education(self, df):

        tasks_dict = dict()
        required_columns = {'age', 'education'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input DataFrame must contain columns {', '.join(required_columns)}.")

        self.n_tasks = len(SPLIT_CRITERIA['age_education'].items())
        for task_name, criteria in SPLIT_CRITERIA['age_education'].items():
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


    def _split_by_age(self, df):
        tasks_dict = dict()
        required_columns = {'age'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input DataFrame must contain columns {', '.join(required_columns)}.")

        self.n_tasks = len(SPLIT_CRITERIA['age'].items())
        for task_name, criteria in SPLIT_CRITERIA['age'].items():
            try:
                task_indices = df.index[df['age'].between(*criteria['age'])].tolist()

                task_df = df.loc[task_indices]
                task_df = task_df.drop(['education', 'age'], axis=1)

                tasks_dict[task_name] = task_df

            except KeyError:
                raise ValueError(
                    f"Invalid criteria structure for task '{task_name}'."
                    f" Ensure 'age' is specified in the criteria."
                )

        return tasks_dict


    def _split_by_n_tasks(self, df):
        tasks_dict = dict()
        num_samples = len(df)
        if self.n_task_samples is None:
            n_samples_per_task = num_samples // self.n_tasks
            remaining_samples =  num_samples % self.n_tasks
        else:
            n_samples_per_task =  self.n_task_samples
            remaining_samples = 0

        start_index = 0

        for i in range(self.n_tasks):
            # Calculate the end index based on the regular number of samples
            end_index = start_index + n_samples_per_task

            # If there are remaining samples, distribute them among the tasks
            if remaining_samples > 0:
                end_index += 1
                remaining_samples -= 1

            task_df = df.iloc[start_index:end_index]
            task_df = task_df.drop(['education', 'age'], axis=1)
            tasks_dict[f"task_{i}"] = task_df

            # Update the start index for the next task
            start_index = end_index

        return tasks_dict


    def _split_by_n_tasks_and_labels(self, df):
        """Split the adult dataset according to the number of tasks, maintaining both the labels in the datasets."""
        tasks_dict = dict()
        df_0 = df[df['income'] == 0]
        df_1 = df[df['income'] == 1]
        n_samples_0 = len(df_0)
        n_samples_1 = len(df_1)
        remaining_df = df.drop(df_0.index).drop(df_1.index)

        if self.n_tasks * self.n_task_samples > len(df):
            raise ValueError("The number of tasks and the number of samples per task are too high for the dataset, "
                             f"which has size {len(df)}."
                             "Please reduce the number of tasks or the number of samples per task.")

        if self.n_task_samples is None:
            n_samples_per_label =  min(n_samples_0, n_samples_1) // self.n_tasks
            remaining_samples =  len(df) - (2 * min(n_samples_0, n_samples_1))
        else:
            n_samples_per_label =  self.n_task_samples // 2
            remaining_samples = self.n_task_samples % 2


        start_index = 0
        remain_index = 0

        for i in range(self.n_tasks):
            end_index = start_index + n_samples_per_label
            task_df = pd.concat([df_0.iloc[start_index:end_index], df_1.iloc[start_index:end_index]])

            if remaining_samples > 0:
                task_df = pd.concat([task_df, remaining_df.iloc[remain_index:remain_index+1]])
                remaining_samples -= 1
                remain_index += 1

            task_df = task_df.drop(['education', 'age'], axis=1)
            tasks_dict[f"task_{i}"] = task_df

            start_index = end_index

        return tasks_dict


    def _split_by_kmeans(self, df):
        """ Split the dataset using k-means"""

        tasks_dict = dict()
        df = df.drop(['education', 'age'], axis=1)
        kmeans = KMeans(n_clusters=self.n_tasks, random_state=self.seed, init='k-means++', n_init='auto')
        clusters = kmeans.fit_predict(df)
        for i in range(self.n_tasks):
            indices = np.where(clusters == i)[0].tolist()
            task_df = df.iloc[indices]
            tasks_dict[f"task_{i}"] = task_df

        return tasks_dict


    def _split_by_gmm(self, df):
        """ Split the dataset using Gaussian Mixture Model"""

        tasks_dict = dict()
        df = df.drop(['education', 'age'], axis=1)
        gmm = GaussianMixture(n_components=self.n_tasks, random_state=self.seed)
        clusters = gmm.fit_predict(df)
        for i in range(self.n_tasks):
            indices = np.where(clusters == i)[0].tolist()
            task_df = df.iloc[indices]
            tasks_dict[f"task_{i}"] = task_df

        return tasks_dict


    def _split_by_prediction(self, df):
        """Split the dataset according to the prediction of a Linear model"""

        columns = df.columns
        dataset = AdultDataset(df)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        device = self.device
        tasks_dict = self._get_model_split(self.split_model, dataloader, columns, device)

        return tasks_dict
    def _split_by_aia(self, df):
        """Split the dataset according to the attribute inference attack"""
        columns = df.columns
        dataset = AdultDataset(df)
        attack_simulator = ModelDrivenAttributeInferenceAttack(model=self.split_model, dataset=dataset,
                                                               device=self.device,
                                                               sensitive_attribute_id=self.sensitive_attribute_id,
                                                               sensitive_attribute_type="binary",
                                                               initialization="normal",
                                                               criterion= nn.BCEWithLogitsLoss(reduction="none").to(self.device),
                                                               is_binary_classification=True,
                                                               learning_rate=0.03,
                                                               optimizer_name="sgd",
                                                               success_metric=threshold_binary_accuracy
                                                  )
        df_dict = attack_simulator.execute_attack_and_split_data()
        tasks_dict = dict()
        df_dict["wrong_reconstructions"].columns = columns
        df_dict["correct_reconstructions"].columns = columns
        tasks_dict["0"] = df_dict["wrong_reconstructions"]
        tasks_dict["1"] = df_dict["correct_reconstructions"]

        return tasks_dict


    def _split_data_into_tasks(self, df):
        """ Split the adult dataset across multiple clients based on specified criteria.

        The available criteria are 'age_education', 'age', 'n_tasks', 'n_tasks_labels', kmeans, gmm, nationality.
        The 'age_education' criterion splits the data based on
        age and education level, while the 'age' criterion splits the data based on age only.

        Args:
         - df (pd.DataFrame):  Input DataFrame containing the columns "age" and "education".
         - split_criterion (str, optional): The criterion used to split the data into tasks. Default is 'age_education'.

        Raises:
            ValueError: If the input DataFrame does not contain the required columns ('age' and 'education').

        Returns:
            - dict: A dictionary where keys are task names and values are DataFrames for each task.
        """
        split_criterion_dict = {
            'age_education': self._split_by_age_education,
            'age': self._split_by_age,
            'n_tasks': self._split_by_n_tasks,
            'n_tasks_labels': self._split_by_n_tasks_and_labels,
            'kmeans': self._split_by_kmeans,
            'gmm': self._split_by_gmm,
            'prediction': self._split_by_prediction,
            'aia': self._split_by_aia
        }

        if self.split_criterion in split_criterion_dict:
            if self.split_criterion in ['n_tasks', 'n_tasks_labels', 'kmeans', 'gmm'] and self.n_tasks is None:
                raise ValueError(f"Number of tasks must be specified when using {', '.join(split_criterion_dict)}' "
                                 "split criteria.")

            if self.split_criterion == 'aia' and self.sensitive_attribute_id is None:
                raise ValueError(f"The sensitive attribute id must be specified when using the 'aia' split criterion.")

            if self.split_criterion == 'n_tasks_labels'and self.n_task_samples < 2:
                raise ValueError(f"The number of samples for each task must be at least 2 when using the "
                                 "'n_tasks_labels' split criterion.")

            tasks_dict = split_criterion_dict[self.split_criterion](df)
        else:
            raise ValueError(f"Invalid criterion '{self.split_criterion}'."
                             f" Supported criteria are {', '.join(split_criterion_dict)}.")

        return tasks_dict


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

    def _save_split_criterion(self):
        with open(self._split_criterion_path, "w") as f:
            criterion_dict = {'split_criterion': self.split_criterion, 'n_tasks': self.n_tasks}
            if self.split_criterion in ['n_tasks', 'n_tasks_labels']:
                criterion_dict['n_task_samples'] = self.n_task_samples
            json.dump(criterion_dict, f)

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

        Remark: We keep the original 'age' and the transformed 'education' columns because they could be needed to split
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

        if self.binarize_marital_status:
            df["marital-status"] = df["marital-status"].apply(self._transform_marital_status)
            CATEGORICAL_COLUMNS.remove('marital-status')

        if self.binarize_race:
            df["race"] = df.race.apply(self._transform_race)
            CATEGORICAL_COLUMNS.remove('race')

        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        if self.test_frac is None:
            train_df, test_df = df[:num_train], df[num_train:]
        else:
            train_df, test_df = train_test_split(df, test_size=self.test_frac, random_state=self.rng.integers(low=0, high=1000))

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
            task_id (int or str): The task number.
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            AdultDataset: An instance of the `AdultDataset` class representing the specified task and data split.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_id = str(task_id)

        task_name = self.task_id_to_name[task_id]
        task_cache_dir = os.path.join(self.cache_dir, 'tasks', self.split_criterion, task_name)
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
