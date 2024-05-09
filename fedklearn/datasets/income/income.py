import os
import shutil
import ssl
import json
import urllib
import logging
from io import StringIO


import torch

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fedklearn.datasets.income.constants import *


class FederatedIncomeDataset:
    """
    A class representing a federated dataset derived from the Income datset.
    This dataset is designed for federated learning scenarios where the data is split across multiple clients,
    and each client represents a specific task based on criteria such as age and education level.

    For more information about the dataset, see: https://www.openml.org/search?type=data&sort=runs&id=43141&status=active
    """

    def __init__(self, cache_dir='./data', download=True, test_frac=0.1, scaler_name="standard", drop_nationality=True,
            rng=None, split_criterion='random', n_tasks=None, n_task_samples=None, force_generation=False,
            seed=42, state='texas', mixing_coefficient=0.):

        self.cache_dir = cache_dir
        self.download = download
        self.test_frac = test_frac
        self.scaler_name = scaler_name
        self.rng = rng
        self.split_criterion = split_criterion
        self.n_tasks = n_tasks
        self.n_task_samples = n_task_samples
        self.force_generation = force_generation
        self.raw_data_dir = os.path.join(self.cache_dir, 'raw')
        self.state=state
        self.drop_nationality = drop_nationality
        self.mixing_coefficient = mixing_coefficient

        if self.state is not None:
            self.intermediate_data_dir = os.path.join(self.cache_dir, 'intermediate', self.state)
            self.tasks_dir = os.path.join(self.cache_dir, 'tasks', self.split_criterion, self.state)
        else:
            self.intermediate_data_dir = os.path.join(self.cache_dir, 'intermediate', 'full')
            self.tasks_dir = os.path.join(self.cache_dir, 'tasks', 'full')

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.seed = seed


        self._metadata_path = os.path.join(self.cache_dir, "metadata.json")

        self._split_criterion_path = os.path.join(self.cache_dir, "split_criterion.json")

        self.scaler = self._set_scaler(self.scaler_name)

        if os.path.exists(self.tasks_dir) and not self.force_generation:
            logging.info(f"Processed data folders found in {self.tasks_dir}. Loading existing files.")
            self._load_task_mapping()

        elif not self.download and self.force_generation:
            if not os.path.exists(self.intermediate_data_dir):
                logging.info(f'Intermediate data not found for state {self.state}. Processing data...')
                self._preprocess()

            logging.info("Data found in the cache directory. Splitting data into tasks..")

            train_df = pd.read_csv(os.path.join(self.intermediate_data_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(self.intermediate_data_dir, "test.csv"))
            self._generate_tasks(train_df, test_df)

        elif  not self.download:
            raise RuntimeError(
                f"Data is not found in {self.raw_data_dir}. Please set `download=True`."
            )

        else:

            # remove the task folder if it exists to avoid inconsistencies
            if os.path.exists(self.tasks_dir):
                shutil.rmtree(self.tasks_dir)

            logging.info("Downloading raw data..")
            os.makedirs(self.raw_data_dir, exist_ok=True)
            self._download_data()

            os.makedirs(self.intermediate_data_dir, exist_ok=True)
            logging.info("Download complete. Processing data..")

            train_df, test_df = self._preprocess()

            self._generate_tasks(train_df, test_df)


    def _download_data(self):
        """Download the raw data from OpenML."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        _, _ = urllib.request.urlretrieve(URL, os.path.join(self.raw_data_dir, "income.arff"))

        logging.info(f"Data downloaded to {self.raw_data_dir}.")


    def _preprocess(self):
        """Preprocess the raw data."""
        data, _ = loadarff(os.path.join(self.raw_data_dir, "income.arff"))
        df = pd.DataFrame(data)

        if self.state is not None:
            if self.state.lower() not in STATES:
                raise ValueError(f"State {self.state} not found in the dataset.")

            df = df[df['ST'] == STATES[self.state]]
            df.drop('ST', axis=1, inplace=True)
            CATEGORICAL_COLUMNS.remove('ST')

        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        if self.drop_nationality:
            df.drop('POBP', axis=1, inplace=True)
            CATEGORICAL_COLUMNS.remove('POBP')

        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        train_df = df.sample(frac=1 - self.test_frac, random_state=self.seed).reset_index(drop=True)
        test_df = df.drop(train_df.index).reset_index(drop=True)

        train_df = self._scale_features(train_df, self.scaler, mode='train')
        test_df = self._scale_features(test_df, self.scaler, mode='test')

        os.makedirs(self.intermediate_data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(self.intermediate_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.intermediate_data_dir, "test.csv"), index=False)
        logging.info(f"Preprocessed data saved to {self.intermediate_data_dir}.")

        return train_df, test_df

    @staticmethod
    def _set_scaler(scaler_name):
        if scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Scaler {scaler_name} not found.")


    @staticmethod
    def _scale_features(df, scaler, mode='train'):

        numerical_columns = df.select_dtypes(include=['number']).columns
        numerical_columns = numerical_columns[numerical_columns != 'PINCP']

        income_col = df['PINCP']

        features_numerical = df[numerical_columns]

        if mode == 'train':
            features_numerical_scaled = \
                pd.DataFrame(scaler.fit_transform(features_numerical), columns=numerical_columns)

        else:
            features_numerical_scaled = \
                pd.DataFrame(scaler.transform(features_numerical), columns=numerical_columns)

        features_scaled = pd.concat([features_numerical_scaled, income_col], axis=1)

        return features_scaled

    def _generate_tasks(self, train_df, test_df):
        """Generate tasks based on the split criterion."""

        logging.info(f"Forcing tasks generation... ")

        if self.split_criterion != 'correlation':
            if os.path.exists(self.tasks_dir):
                shutil.rmtree(self.tasks_dir)

        train_tasks_dict = self._split_data_into_tasks(train_df)
        test_tasks_dict = self._split_data_into_tasks(test_df)

        task_dicts = [train_tasks_dict, test_tasks_dict]

        self.task_id_to_name = {f'{i}': task_name for i, task_name in enumerate(train_tasks_dict.keys())}

        for mode, task_dict in zip(['train', 'test'], task_dicts):
            for task_name, task_data in task_dict.items():
                if self.split_criterion == 'correlation':
                    task_cache_dir = os.path.join(self.tasks_dir, f'{int(self.mixing_coefficient * 100)}',task_name)
                else:
                    task_cache_dir = os.path.join(self.tasks_dir,task_name)
                os.makedirs(task_cache_dir, exist_ok=True)

                file_path = os.path.join(task_cache_dir, f"{mode}.csv")
                task_data.to_csv(file_path, index=False)

        logging.info(f"Tasks generated and saved to {self.tasks_dir}.")

        self._save_task_mapping()

        self._save_split_criterion()


    def _save_task_mapping(self):
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r") as f:
                metadata = json.load(f)
                metadata[self.split_criterion] = self.task_id_to_name
            with open(self._metadata_path, "w") as f:
                json.dump(metadata, f)
        else:
            with open(self._metadata_path, "w") as f:
                metadata = {self.split_criterion: self.task_id_to_name}
                json.dump(metadata, f)


    def _load_task_mapping(self):
        with (open(self._metadata_path, "r") as f):
            metadata = json.load(f)
            self.task_id_to_name = metadata[self.split_criterion]


    def _save_split_criterion(self):
        criterion_dict = {'split_criterion': self.split_criterion}
        if self.split_criterion in ['correlation']:
            criterion_dict['n_task_samples'] = self.n_task_samples
        with open(self._split_criterion_path, "w") as f:
            json.dump(criterion_dict, f)


    def _iid_tasks_divide(self, df, n_tasks):
        """
        Split a dataframe into a dictionary of dataframes.
        Args:
            df(pd.DataFrame): DataFrame to split into tasks.

        Returns:
            tasks_dict(Dict[str, pd.DataFrame]): A dictionary mapping task IDs to dataframes.

        """
        num_elems = len(df)
        group_size = int(len(df) // n_tasks)
        num_big_groups = num_elems - (n_tasks * group_size)
        num_small_groups = n_tasks - num_big_groups
        tasks_dict = dict()

        for i in range(num_small_groups):
            tasks_dict[f"{i}"] = df.iloc[group_size * i: group_size * (i + 1)]
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            tasks_dict[f"{i + num_small_groups}"] = df.iloc[bi + group_size * i:bi + group_size * (i + 1)]


        return tasks_dict

    def _split_by_correlation(self, df):

        lower_bound = min(df['SEX'])
        upper_bound = max(df['SEX'])

        median_income = df['PINCP'].median()

        df_rich_men_poor_women = df[(((df['PINCP'] > median_income) & (df['SEX'] == lower_bound)) |
                                    ((df['PINCP'] <= median_income) & (df['SEX'] == upper_bound)))]
        df_poor_men_rich_women = df.drop(df_rich_men_poor_women.index)


        if self.mixing_coefficient < 0 or self.mixing_coefficient > 1:
            raise ValueError("The mixing coefficient must be between 0 and 1.")

        if self.mixing_coefficient > 0:
            n_mix_samples_rmpw = int(self.mixing_coefficient * len(df_rich_men_poor_women))
            n_mix_samples_pmrw = int(self.mixing_coefficient * len(df_poor_men_rich_women))
            mix_sample_rich_men_poor_women = df_rich_men_poor_women.sample(n=n_mix_samples_pmrw)
            mix_sample_poor_men_rich_women = df_poor_men_rich_women.sample(n=n_mix_samples_rmpw)

            df_rich_men_poor_women = df_rich_men_poor_women[n_mix_samples_rmpw:]
            df_poor_men_rich_women = df_poor_men_rich_women[n_mix_samples_pmrw:]

            df_rich_men_poor_women = pd.concat([df_rich_men_poor_women, mix_sample_poor_men_rich_women], axis=0)
            df_poor_men_rich_women = pd.concat([df_poor_men_rich_women, mix_sample_rich_men_poor_women], axis=0)

            # shuffle the data
            df_rich_men_poor_women = df_rich_men_poor_women.sample(frac=1, random_state=self.seed)
            df_poor_men_rich_women = df_poor_men_rich_women.sample(frac=1, random_state=self.seed)

        if self.n_task_samples is None:
            tasks_dict_poor_men = self._iid_tasks_divide(df_poor_men_rich_women, self.n_tasks // 2)
            if self.n_tasks % 2 != 0:
                tasks_dict_rich_men = self._iid_tasks_divide(df_rich_men_poor_women, self.n_tasks // 2 + 1)
            else:
                tasks_dict_rich_men = self._iid_tasks_divide(df_rich_men_poor_women, self.n_tasks // 2)

            tasks_dict_rich_men = {str(int(k) + self.n_tasks // 2): v for k, v in tasks_dict_rich_men.items()}
            tasks_dict = {**tasks_dict_poor_men, **tasks_dict_rich_men}

        elif self.n_tasks * self.n_task_samples > len(df):
                raise ValueError("The number of tasks and the number of samples per task are too high for the dataset, "
                             f"which has size {len(df)}."
                             "Please reduce the number of tasks or the number of samples per task.")
        else:
            tasks_dict_rich_men = dict()
            tasks_dict_poor_men = dict()
            for i in range(self.n_tasks // 2):
                tasks_dict_poor_men[f"{i}"] = df_poor_men_rich_women.iloc[i * self.n_task_samples:(i + 1) * self.n_task_samples]
                tasks_dict_rich_men[f"{i}"] = df_rich_men_poor_women[i * self.n_task_samples:(i + 1) * self.n_task_samples]

            if self.n_tasks % 2 != 0:
                tasks_dict_rich_men[f"{self.n_tasks // 2}"] = df_rich_men_poor_women[self.n_tasks // 2 * self.n_task_samples:
                                                                           self.n_tasks // 2 * self.n_task_samples +
                                                                           self.n_task_samples]
            tasks_dict_rich_men = {str(int(k) + self.n_tasks // 2): v for k, v in tasks_dict_rich_men.items()}

            tasks_dict = {**tasks_dict_poor_men, **tasks_dict_rich_men}

        return tasks_dict


    def _random_split(self, df):
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


    def _split_data_into_tasks(self, df):
        split_criterion_dict = {
            'random': self._random_split,
            'correlation': self._split_by_correlation,
        }
        if self.split_criterion not in split_criterion_dict:
            raise ValueError(f"Invalid split critrion. Supported criteria are {', '.join(split_criterion_dict)}.")
        else:
            return split_criterion_dict[self.split_criterion](df)


    def get_task_dataset(self, task_id, mode='train'):

        task_id = f'{task_id}'

        if mode not in {'train', 'test'}:
            raise ValueError(f"Mode '{mode}' is not recognized.  Supported values are 'train' or 'test'.")

        task_name = self.task_id_to_name[task_id]
        if self.split_criterion == 'correlation':
            file_path = os.path.join(self.tasks_dir, f'{int(self.mixing_coefficient * 100)}', task_name, f"{mode}.csv")
        else:
            file_path = os.path.join(self.tasks_dir, task_name, f"{mode}.csv")
        task_data = pd.read_csv(file_path)

        return IncomeDataset(task_data, name=task_name)


    def get_pooled_data(self, mode="train"):
        """
        Returns the pooled dataset before splitting into tasks.

        Args:
            mode (str, optional): The type of data split, either 'train' or 'test'. Default is 'train'.

        Returns:
            IncomeDataset: An instance of the `IncomeDataset` class containing the pooled data.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        file_path = os.path.join(self.intermediate_data_dir, f'{mode}.csv')

        data = pd.read_csv(file_path)

        return AdultDataset(data, name="pooled")


class IncomeDataset(Dataset):

    def __init__(self, dataframe, name=None):

        self.column_names = list(dataframe.columns.drop('PINCP'))
        self.column_name_to_id = {name: i for i, name in enumerate(self.column_names)}

        self.features = dataframe.drop('PINCP', axis=1).values
        self.targets = dataframe['PINCP'].values

        self.name = name

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), np.float32(self.targets[idx])


if __name__ == "__main__":
    dataset = FederatedIncomeDataset(cache_dir="../../../scripts/data/income", download=True,
                                     test_frac=0.1, scaler_name="standard", drop_nationality=True,
                                     rng=None, split_criterion='correlation', n_tasks=10, force_generation=True,
                                     seed=42, state='nevada', mixing_coefficient=0.)
