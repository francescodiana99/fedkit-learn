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
    and each client represents a specific task based on criteria such as age USA state.

    For more information about the dataset, see: https://www.openml.org/search?type=data&sort=runs&id=43141&status=active

    Args:
        cache_dir (str, optional): The directory path for caching downloaded and preprocessed data. Default is "./".

        download (bool, optional): Whether to download the raw data. Default is True.

        test_frac (float, optional): Fraction of the test samples; it should be a float between 0 and 1.

        scaler_name (str, optional): The name of the scaler to use for feature scaling. Default is "standard".

        rng (Random Number Generator, optional): An instance of a random number generator.
            If `None`, a new generator will be created. Default is `None`.

        split_criterion (str, optional): The criterion used to split the data into tasks. Default is "random".

        n_tasks (int, optional): The number of tasks to split the data into. Default is `None`.

        n_task_samples (int, optional): The number of samples per task.
            If 'None',all the data will be used. Default is `None`.

        force_generation (bool, optional): Whether to force the generation of tasks from the pre-processed data
            even if they already exist. Default is False.

        seed (int, optional): The seed used for random number generation. Default is 42.

        state (str, optional): USA state data to use.
            If 'full', the full dataset will be used. Default is full.

        mixing_coefficient (float, optional): Mixing coefficient used to manage the correlation between 'SEX' and target
            variable, when using  'correlation' as split criterion. Default is 0.

    Attributes:
        cache_dir (str): The directory path for caching downloaded and preprocessed data.

        download (bool): Whether to download the raw data.

        test_frac (float): Fraction of the test samples.

        scaler_name (str): The name of the scaler to use for feature scaling.

        rng (Random Number Generator): An instance of a random number generator.

        split_criterion (str): The criterion used to split the data into tasks. Available options are 'random', 'state',
            and 'correlation'. Default is 'random'.

        n_tasks (int): The number of tasks to split the data into.

        n_task_samples (int): The number of samples per task.

        force_generation (bool): Whether to force the generation of tasks from the pre-processed data
            even if they already exist.

        _raw_data_dir (str): The directory path for the raw data.

        _intermediate_data_dir (str): The directory path for the preprocessed data.

        _tasks_dir (str): The directory path for the tasks.

        state (str): USA state data to use. If 'full', the full dataset will be used. Default is 'full'.

        drop_nationality (bool): Whether to drop the 'POBP' column from the dataset.

        mixing_coefficient (float): Mixing coefficient used to manage the correlation.

        _metadata_path (str): The path to the metadata file.

        _split_criterion_path (str): The path to the split criterion file.

        scaler (sklearn.preprocessing): The scaler used for feature scaling.

        task_id_to_name (dict): A dictionary mapping task IDs to task names.

    Methods:
        _download_data: Download the raw data from OpenML.

        _preprocess: Preprocess the raw data when using a specific state.

        _preprocess_full_data: Preprocess the raw data when using the full dataset.

        _sample_state: Sample data from a DataFrame group.

        _generate_tasks: Generate tasks based on the split criterion.

        _set_scaler: Set the scaler for feature scaling.

        _scale_features: Scale the features using the specified scaler.

        _save_task_mapping: Save the task mapping to the metadata file.

        _load_task_mapping: Load the task mapping from the metadata file.

        _save_split_criterion: Save the split criterion to the split criterion file.

        _iid_tasks_divide: Split a dataframe into a dictionary of dataframes.

        _split_by_correlation: Split the data based on the correlation between the target variable and the 'SEX' column.

        _random_split: Split the data randomly into tasks.

        _split_by_state: Split the data based on the USA state.

        _split_data_into_tasks: Split the data into tasks based on the split criterion.

        get_task_dataset: Get the dataset for a specific task.

        get_pooled_data: Get tall the data before splitting into tasks.

    """

    def __init__(self, cache_dir='./', download=True, test_frac=0.1, scaler_name="standard", drop_nationality=True,
            rng=None, split_criterion='random', n_tasks=None, n_task_samples=None, force_generation=False,
            seed=42, state='full', mixing_coefficient=0.):

        self.cache_dir = cache_dir
        self.download = download
        self.test_frac = test_frac
        self.scaler_name = scaler_name
        self.rng = rng
        self.split_criterion = split_criterion
        self.n_tasks = n_tasks
        self.n_task_samples = n_task_samples
        self.force_generation = force_generation
        self._raw_data_dir = os.path.join(self.cache_dir, 'raw')
        self.state=state
        self.drop_nationality = drop_nationality
        self.mixing_coefficient = mixing_coefficient

        if self.state is None:
            raise ValueError("The 'state' is None. Please specify a value.")


        self._intermediate_data_dir = os.path.join(self.cache_dir, 'intermediate', self.state)
        self._tasks_dir = os.path.join(self.cache_dir, 'tasks', self.split_criterion, self.state)

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.seed = seed


        self._metadata_path = os.path.join(self.cache_dir, "metadata.json")

        if self.split_criterion == 'correlation':
            if self.n_task_samples is None:
                self._split_criterion_path = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}',
                                              f'{self.n_tasks}', 'all', 'split_criterion.json')
            else:
                self._split_criterion_path = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}',
                                              f'{self.n_tasks}', f'{self.n_task_samples}', 'split_criterion.json')
        else:
            if self.n_task_samples is None:
                self._split_criterion_path = os.path.join(self._tasks_dir, f'{self.n_tasks}', 'all', 'split_criterion.json')
            else:
                self._split_criterion_path = os.path.join(self._tasks_dir, f'{self.n_tasks}', f'{self.n_task_samples}', 'split_criterion.json')


        self.scaler = self._set_scaler(self.scaler_name)
        print(self._tasks_dir)
        print(os.path.exists(self._tasks_dir))
        print(self.force_generation
        if os.path.exists(self._tasks_dir) and not self.force_generation:
            logging.info(f"Processed data folders found in {self._tasks_dir}. Loading existing files.")
            self._load_task_mapping()

        elif not self.download and self.force_generation:
            if state == 'full':
                logging.info(f'Using full split criterion. Processing data..')
                self._preprocess_full_data()
            else:
                if not os.path.exists(self._intermediate_data_dir):
                    logging.info(f'Intermediate data not found for state {self.state}. Processing data...')
                    self._preprocess()

            logging.info("Data found in the cache directory. Splitting data into tasks..")

            train_df = pd.read_csv(os.path.join(self._intermediate_data_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(self._intermediate_data_dir, "test.csv"))
            self._generate_tasks(train_df, test_df)

        elif  not self.download:
            raise RuntimeError(
                f"Data is not found in {self._tasks_dir}. Please set `download=True`."
            )

        else:

            # remove the task folder if it exists to avoid inconsistencies
            if os.path.exists(self._tasks_dir):
                shutil.rmtree(self._tasks_dir)

            logging.info("Downloading raw data..")
            os.makedirs(self._raw_data_dir, exist_ok=True)
            self._download_data()

            os.makedirs(self._intermediate_data_dir, exist_ok=True)
            logging.info("Download complete. Processing data..")
            if self.state == 'full':
               train_df, test_df = self._preprocess_full_data()
            else:
               train_df, test_df =  self._preprocess()

            self._generate_tasks(train_df, test_df)


    def _download_data(self):
        """Download the raw data from OpenML."""
        os.makedirs(self._raw_data_dir, exist_ok=True)
        _, _ = urllib.request.urlretrieve(URL, os.path.join(self._raw_data_dir, "income.arff"))

        logging.info(f"Data downloaded to {self._raw_data_dir}.")
        data, _ = loadarff(os.path.join(self._raw_data_dir, "income.arff"))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self._raw_data_dir, "income.csv"), index=False)


    def _preprocess(self):
        """Prepare the raw data for splitting in tasks."""
        df = pd.read_csv(os.path.join(self._raw_data_dir, "income.csv"))

        if self.state.lower() not in STATES:
            raise ValueError(f"State {self.state} not found in the dataset.")

        df = df[df['ST'] == STATES[self.state]]

        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        if self.drop_nationality:
            df.drop('POBP', axis=1, inplace=True)
            CATEGORICAL_COLUMNS.remove('POBP')

        df.drop('ST', axis=1, inplace=True)
        CATEGORICAL_COLUMNS.remove('ST')

        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        train_df = df.sample(frac=1 - self.test_frac, random_state=self.seed)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        train_df.reset_index(drop=True, inplace=True)

        train_df = self._scale_features(train_df, self.scaler, mode='train')
        test_df = self._scale_features(test_df, self.scaler, mode='test')

        os.makedirs(self._intermediate_data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(self._intermediate_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self._intermediate_data_dir, "test.csv"), index=False)

        logging.info(f"Preprocessed data saved to {self._intermediate_data_dir}.")

        return train_df, test_df


    @staticmethod
    def _sample_state(state_group, n_samples):
        """Sample data from a DataFrame group.
        Args:
            state_group(pd.DataFrame): The DataFrame group to sample from.
            n_samples(int): The number of samples to draw."""
        indices = np.random.choice(state_group.index, n_samples, replace=False)
        return state_group.loc[indices]

    def _preprocess_full_data(self):
        """
        Prepare the raw data for scaling when using the full data.

        Returns:
            pd.DataFrame: Sampled and preocessed training data.

        """

        df = pd.read_csv(os.path.join(self._raw_data_dir, "income.csv"))

        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        df_grouped = df.groupby('ST')
        state_samples = df_grouped.size().sort_values(ascending=False)

        if any(self.n_task_samples > state_samples.head(self.n_tasks)):
            raise ValueError(f"The number of samples per task must be less than or equal to the number of samples in "
                             f"the smallest state considered which is {state_samples.iloc[:self.n_tasks]}.")
        if self.n_tasks > len(state_samples):
            raise ValueError(f"The number of tasks must be less than or equal to the number of states, "
                             f"which is {len(state_samples)}.")

        df_filtered = df[df['ST'].isin(state_samples.head(self.n_tasks).index[:self.n_tasks])]

        sampled_states = [self._sample_state(state_group, self.n_task_samples) for state, state_group in df_filtered.groupby('ST')]
        df = pd.concat(sampled_states).reset_index(drop=True)

        if self.drop_nationality:
            df.drop('POBP', axis=1, inplace=True)
            CATEGORICAL_COLUMNS.remove('POBP')

        state_col = df['ST']
        df.drop('ST', axis=1, inplace=True)
        CATEGORICAL_COLUMNS.remove('ST')
        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)

        df['ST'] = state_col
        CATEGORICAL_COLUMNS.append('ST')

        df_grouped = df.groupby('ST')
        n_train = int(df_grouped.size().min() * (1 - self.test_frac))
        sampled_states = [self._sample_state(state_group, n_train) for state, state_group in df_grouped]
        train_df = pd.concat(sampled_states)

        test_df = df.drop(train_df.index).reset_index(drop=True)
        train_df.reset_index(drop=True, inplace=True)

        train_df = self._scale_features(train_df, self.scaler, mode='train')
        test_df = self._scale_features(test_df, self.scaler, mode='test')

        os.makedirs(self._intermediate_data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(self._intermediate_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self._intermediate_data_dir, "test.csv"), index=False)

        logging.info(f"Preprocessed data saved to {self._intermediate_data_dir}.")

        return train_df, test_df

    @staticmethod
    def _set_scaler(scaler_name):
        if scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Scaler {scaler_name} not found.")

    def _scale_features(self, df, scaler, mode='train'):

        dummy_columns = df.select_dtypes(include=['Sparse']).columns
        dummy_columns = list(set(dummy_columns) - set(NON_CATEGORICAL_COLUMNS))
        if 'ST' in dummy_columns:
            dummy_columns.remove('ST')

        if self.state == 'full':
            state_col = df['ST']
        income_col = df['PINCP']

        features_numerical = df[NON_CATEGORICAL_COLUMNS]
        features_numerical = features_numerical.drop('PINCP', axis=1)

        features_dummy = df[dummy_columns]

        numerical_columns = features_numerical.columns

        if mode == 'train':
            features_numerical_scaled = \
                pd.DataFrame(scaler.fit_transform(features_numerical), columns=numerical_columns)

        else:
            features_numerical_scaled = \
                pd.DataFrame(scaler.transform(features_numerical), columns=numerical_columns)

        if self.state == 'full':
            features_scaled = pd.concat([features_dummy, state_col, features_numerical_scaled, income_col], axis=1)
        else:
            features_scaled = pd.concat([features_dummy, features_numerical_scaled, income_col], axis=1)

        return features_scaled

    def _generate_tasks(self, train_df, test_df):
        """Generate tasks based on the split criterion."""

        logging.info(f"Forcing tasks generation... ")

        if self.split_criterion != 'correlation':
            if os.path.exists(self._tasks_dir):
                shutil.rmtree(self._tasks_dir)

        train_tasks_dict = self._split_data_into_tasks(train_df, mode='train')
        test_tasks_dict = self._split_data_into_tasks(test_df, mode='test')

        task_dicts = [train_tasks_dict, test_tasks_dict]

        self.task_id_to_name = {f'{i}': task_name for i, task_name in enumerate(train_tasks_dict.keys())}

        for mode, task_dict in zip(['train', 'test'], task_dicts):
            for task_name, task_data in task_dict.items():
                task_cache_dir = self._get_task_cache_dir(task_name)

                os.makedirs(task_cache_dir, exist_ok=True)

                file_path = os.path.join(task_cache_dir, f"{mode}.csv")
                task_data.to_csv(file_path, index=False)
                # TODO: this should be not in the loop, and coded better

        logging.info(f"Tasks generated and saved to {self._tasks_dir}.")

        self._save_task_mapping()

        self._save_split_criterion()

    def _get_task_cache_dir(self, task_name):
        if self.split_criterion == 'correlation':
            if self.n_task_samples is None:
                task_cache_dir = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}',
                                              f'{self.n_tasks}', 'all', task_name)
            else:
                task_cache_dir = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}',
                                              f'{self.n_tasks}', f'{self.n_task_samples}', task_name)
        else:
            if self.n_task_samples is None:
                task_cache_dir = os.path.join(self._tasks_dir, f'{self.n_tasks}', 'all', task_name)
            else:
                task_cache_dir = os.path.join(self._tasks_dir, f'{self.n_tasks}', f'{self.n_task_samples}', task_name)

        return task_cache_dir

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
        criterion_dict = {'split_criterion': self.split_criterion,
                          'cache_dir': self.cache_dir,
                          'n_tasks': self.n_tasks,
                          'n_task_samples':self.n_task_samples
                          }

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

    def _split_by_correlation(self, df, mode='train'):

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
            mix_sample_rich_men_poor_women = df_rich_men_poor_women.sample(n=n_mix_samples_pmrw, random_state=self.seed)
            mix_sample_poor_men_rich_women = df_poor_men_rich_women.sample(n=n_mix_samples_rmpw, random_state=self.seed)

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

        else:
            n_train_samples = self.n_task_samples * (1 - self.test_frac)
            if self.n_tasks * n_train_samples > len(df) and mode == 'train':
                raise ValueError("The number of tasks and the number of samples per task are too high for the dataset, "
                                 f"which has size {len(df)}."
                                 "Please reduce the number of tasks or the number of samples per task.")
            elif self.n_tasks * self.n_task_samples * self.test_frac > len(df) and mode == 'test':
                raise ValueError("The number of tasks and the number of samples per task are too high for the dataset, "
                                 f"which has size {len(df)}."
                                 "Please reduce the number of tasks or the number of samples per task.")

            tasks_dict_rich_men = dict()
            tasks_dict_poor_men = dict()

            if mode == 'train':
                for i in range(self.n_tasks // 2):
                    tasks_dict_poor_men[f"{i}"] = df_poor_men_rich_women.iloc[i * n_train_samples:(i + 1) * n_train_samples]
                    tasks_dict_rich_men[f"{i}"] = df_rich_men_poor_women[i * n_train_samples:(i + 1) * n_train_samples]

                if self.n_tasks % 2 != 0:
                    tasks_dict_rich_men[f"{self.n_tasks // 2}"] = df_rich_men_poor_women[self.n_tasks // 2 * n_train_samples:
                                                                               self.n_tasks // 2 * self.n_task_samples +
                                                                               self.n_task_samples]
            else:
                n_test_samples = int(self.n_task_samples * self.test_frac)
                for i in range(self.n_tasks // 2):
                    tasks_dict_poor_men[f"{i}"] = df_poor_men_rich_women.iloc[
                                                  i * n_test_samples:(i + 1) * n_test_samples]
                    tasks_dict_rich_men[f"{i}"] = df_rich_men_poor_women[
                                                  i * n_test_samples:(i + 1) * n_test_samples]

                if self.n_tasks % 2 != 0:
                    tasks_dict_rich_men[f"{self.n_tasks // 2}"] = df_rich_men_poor_women[
                                                                  self.n_tasks // 2 * n_test_samples:
                                                                  self.n_tasks // 2 * n_test_samples +
                                                                  n_test_samples]

            tasks_dict_rich_men = {str(int(k) + self.n_tasks // 2): v for k, v in tasks_dict_rich_men.items()}

            tasks_dict = {**tasks_dict_poor_men, **tasks_dict_rich_men}

        return tasks_dict


    def _random_split(self, df):

        num_elems = len(df)
        group_size = int(len(df) // self.n_tasks)
        num_big_groups = num_elems - self.n_tasks * group_size
        num_small_groups = self.n_tasks - num_big_groups
        tasks_dict = dict()

        df.drop('ST', axis=1, inplace=True)

        for i in range(num_small_groups):
            tasks_dict[f"{i}"] = df.iloc[group_size * i: group_size * (i + 1)]
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            tasks_dict[f"{i + num_small_groups}"] = df.iloc[bi + group_size * i:bi + group_size * (i + 1)]

        return tasks_dict


    def _split_by_income_sex(self, df):
        if self.n_tasks is None:
            raise ValueError("The number of tasks must be specified")

        men = df['SEX'].min()
        women = df['SEX'].max()

        percentiles = [i/self.n_tasks for i in range(0,self.n_tasks + 1)]
        income_percentiles = list(df['PINCP'].quantile(percentiles))

        tasks_dict = dict()

        for i in range(self.n_tasks):
            df_men = df[(df['SEX'] == men) & (income_percentiles[i] < df['PINCP'] ) & (df['PINCP']  <= income_percentiles[i+1])]
            df_women = df[(df['SEX'] == women) & (income_percentiles[i] < df['PINCP'] ) & (df['PINCP']  <= income_percentiles[i+1])]

            if len(df_men) < len(df_women):
               df_women =  df_women.sample(n=len(df_men), random_state=self.seed)
            else:
                df_men = df_men.sample(n=len(df_women), random_state=self.seed)

            tasks_dict[f"{i}"] = pd.concat([df_men, df_women], axis=0)

        return tasks_dict

    def _split_by_state(self, df, mode='train'):
        if self.state != 'full':
            raise ValueError("The state split criterion is supported only for the full dataset. ")
        if self.n_tasks is None:
            raise ValueError("The number of tasks must be specified for the state split criterion.")

        if self.n_tasks > len(STATES):
            raise ValueError(f"The number of tasks must be less than or equal to the number of states, "
                             f"which is {len(STATES)}.")

        tasks_dict = dict()
        states_list = [k for k, v in df['ST'].value_counts().items()]
        if mode == 'train':
            n_samples = int(self.n_task_samples * (1 - self.test_frac))
        else:
            n_samples = int(self.n_task_samples * self.test_frac)

        for i in range(self.n_tasks):
            state_code = states_list[i]
            tasks_dict[f"{i}"] = df[df['ST'] == state_code].sample(n=n_samples, random_state=self.seed).copy()
            tasks_dict[f"{i}"].drop('ST', axis=1, inplace=True)

        return tasks_dict

    def _split_data_into_tasks(self, df, mode='train'):
        split_criterion_dict = {
            'random': self._random_split,
            'correlation': self._split_by_correlation,
            'state': self._split_by_state,
            'income_sex': self._split_by_income_sex
        }
        if self.split_criterion not in split_criterion_dict:
            raise ValueError(f"Invalid split critrion. Supported criteria are {', '.join(split_criterion_dict)}.")
        if self.state == 'full' or self.split_criterion == 'correlation':
            return split_criterion_dict[self.split_criterion](df, mode=mode)
        else:
            return split_criterion_dict[self.split_criterion](df)


    def get_task_dataset(self, task_id, mode='train'):

        task_id = f'{task_id}'

        if mode not in {'train', 'test'}:
            raise ValueError(f"Mode '{mode}' is not recognized.  Supported values are 'train' or 'test'.")

        task_name = self.task_id_to_name[task_id]
        if self.split_criterion == 'correlation':
            if self.n_task_samples is None:
                file_path = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}', f'{self.n_tasks}',
                                         'all', task_name, f"{mode}.csv")
            else:
                file_path = os.path.join(self._tasks_dir, f'{int(self.mixing_coefficient * 100)}', f'{self.n_tasks}',
                                         f'{self.n_task_samples}',task_name, f"{mode}.csv")

        else:
            if self.n_task_samples is None:
                file_path = os.path.join(self._tasks_dir, f'{self.n_tasks}', 'all', task_name, f"{mode}.csv")
            else:
                file_path = os.path.join(self._tasks_dir, f'{self.n_tasks}',  f'{self.n_task_samples}'  , task_name,
                                         f"{mode}.csv")
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

        file_path = os.path.join(self._intermediate_data_dir, f'{mode}.csv')

        data = pd.read_csv(file_path)

        return IncomeDataset(data, name="pooled")


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
