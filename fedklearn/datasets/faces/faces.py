import os
import logging
import requests
import zipfile
import imageio

import torch
from torch.utils.data import TensorDataset

import numpy as np

import matplotlib.pyplot as plt

from .constants import *

from ..utils import iid_divide


class FederatedFacesDataset:
    def __init__(
            self, cache_dir="./", n_tasks=None, n_classes_per_task=None, test_frac=None, download=True,
            force_generation=True, rng=None
    ):
        self.cache_dir = cache_dir
        self.download = download
        self.force_generation = force_generation

        self.raw_data_dir = os.path.join(self.cache_dir, "raw")
        self.intermediate_data_dir = os.path.join(self.cache_dir, "intermediate")
        self.tasks_dir = os.path.join(self.cache_dir, "tasks")

        self.rng = rng if rng is not None else np.random.default_rng()

        if os.path.exists(self.intermediate_data_dir):
            logging.info("Intermediate data folders found. Loading existing files..")
            self.all_data = self._load_all_images()

        elif self.download:
            logging.info("==> Downloading raw data..")
            os.makedirs(self.raw_data_dir, exist_ok=True)
            self._download()

            logging.info(f"==> Reading and saving images..")
            os.makedirs(self.intermediate_data_dir, exist_ok=True)
            self.all_data = self._preprocess()

        else:
            raise RuntimeError(
                f"Data is not found in {self.intermediate_data_dir}. Please set `download=True`."
            )

        if os.path.exists(self.tasks_dir) and not self.force_generation:
            logging.info("Processed data folders found. Loading existing files..")

        else:
            self.n_tasks = n_tasks
            self.n_classes_per_task = n_classes_per_task

            self.test_frac = test_frac if test_frac is not None else 0.

            self.n_shards_per_subject = max(1, (self.n_classes_per_task * self.n_tasks) // N_SUBJECTS)

            self._shards_per_subject_dict = self._get_subject_shards()

            self.train_images_per_task_dict, self.train_labels_per_task_dict, \
                self.test_images_per_task_dict, self.test_labels_per_task_dict = self._split_data_into_tasks()

            logging.info("==> Generating data splits..")
            os.makedirs(self.tasks_dir, exist_ok=True)

            for task_id in range(self.n_tasks):
                train_images = self.train_images_per_task_dict[task_id]
                train_labels = self.train_labels_per_task_dict[task_id]

                test_images = self.test_images_per_task_dict[task_id]
                test_labels = self.test_labels_per_task_dict[task_id]

                task_dir = os.path.join(self.tasks_dir, f"{task_id}")

                train_save_path = os.path.join(task_dir, "train.npz")
                test_save_path = os.path.join(task_dir, "test.npz")

                os.makedirs(task_dir, exist_ok=True)
                np.savez_compressed(train_save_path, images=train_images, labels=train_labels)
                np.savez_compressed(test_save_path, images=test_images, labels=test_labels)

            logging.info("data saved successfully.")

    @staticmethod
    def _download_file(url, output_filename):
        response = requests.get(url, stream=True)
        with open(output_filename, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                if data:
                    file.write(data)

    @staticmethod
    def _unzip_file(zipfile_path, output_dir):
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    @staticmethod
    def _read_pgm(file_path):
        try:
            img = imageio.imread(file_path, format='PGM')
            return img
        except Exception as e:
            print(f"Error reading PGM image: {e}")
            return None

    @staticmethod
    def visualize_images(images_array, n_cols=5):
        num_images = images_array.shape[0]

        n_rows = num_images // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)

        for i in range(n_rows):
            for j in range(n_cols):
                index = i * n_cols + j

                if index < num_images:
                    ax = axes[i, j]
                    ax.imshow(images_array[index], cmap='gray')
                    ax.set_title(f"Image {index + 1}")
                    ax.axis("off")
                else:
                    axes[i, j].axis("off")

        plt.show()

    def _download(self):
        zipfile_path = os.path.join(self.cache_dir, "raw", ZIP_FILENAME)
        self._download_file(url=URL, output_filename=zipfile_path)
        logging.debug(f"==> Raw data downloaded to {zipfile_path}.")

        self._unzip_file(zipfile_path=zipfile_path, output_dir=self.raw_data_dir)
        logging.debug(f"==> Raw data extracted to {self.raw_data_dir}.")

    def _read_images_dir(self, images_dir):
        images = []

        for image_id in os.listdir(images_dir):
            img_path = os.path.join(images_dir, image_id)

            img = self._read_pgm(img_path)

            images.append(img)

        images = np.stack(images)

        return images

    def _preprocess(self):
        subjects_dir = os.path.join(self.raw_data_dir, ZIP_FILENAME.split(".")[0])

        all_data = []
        for subject_id in os.listdir(subjects_dir):
            if subject_id == "README":
                continue

            images_dir = os.path.join(subjects_dir, subject_id)
            save_path = os.path.join(self.intermediate_data_dir, f"{subject_id}.npy")
            subject_data = self._read_images_dir(images_dir=images_dir)
            np.save(save_path, subject_data)

            all_data.append(subject_data)

        all_data = np.stack(all_data)
        logging.info(f"==> Images saved to {self.intermediate_data_dir}")

        return all_data

    def _load_all_images(self):
        all_data = []
        for subject_file in os.listdir(self.intermediate_data_dir):
            subject_data_path = os.path.join(self.intermediate_data_dir, subject_file)
            subject_data = np.load(subject_data_path)

            all_data.append(subject_data)

        all_data = np.stack(all_data)

        return all_data

    def _get_subject_shards(self):
        shards_per_subject_dict = dict()

        for subject_id in range(N_SUBJECTS):
            n_samples = len(self.all_data[subject_id])

            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            shards_per_subject_dict[subject_id] = iid_divide(indices, self.n_shards_per_subject)

        return shards_per_subject_dict

    def _split_data_into_tasks(self):
        task_id = 0

        images_per_task_dict = dict()
        labels_per_task_dict = dict()

        shuffled_subject_ids = self.rng.permutation(N_SUBJECTS)

        for subject_id in shuffled_subject_ids:
            for shard in self._shards_per_subject_dict[subject_id]:
                c_data = self.all_data[subject_id][shard]

                if task_id in images_per_task_dict:
                    images_per_task_dict[task_id].append(c_data)
                    labels_per_task_dict[task_id].append(np.full(shape=len(c_data), fill_value=subject_id))

                else:
                    images_per_task_dict[task_id] = [c_data]
                    labels_per_task_dict[task_id] = [np.full(shape=len(c_data), fill_value=subject_id)]

                task_id += 1
                task_id = task_id % self.n_tasks

        # train test split
        train_images_per_task_dict = dict()
        test_images_per_task_dict = dict()
        train_labels_per_task_dict = dict()
        test_labels_per_task_dict = dict()

        for task_id in images_per_task_dict:
            images_per_task_dict[task_id] = np.concatenate(images_per_task_dict[task_id])
            labels_per_task_dict[task_id] = np.concatenate(labels_per_task_dict[task_id])

            n_samples = len(labels_per_task_dict[task_id])
            n_test_samples = int(n_samples * self.test_frac)
            n_train_samples = n_samples - n_test_samples

            all_indices = self.rng.permutation(n_samples)
            train_indices = all_indices[:n_train_samples]
            test_indices = all_indices[n_train_samples:]

            train_images_per_task_dict[task_id], test_images_per_task_dict[task_id] = (
                images_per_task_dict[task_id][train_indices],
                images_per_task_dict[task_id][test_indices]
            )

            train_labels_per_task_dict[task_id], test_labels_per_task_dict[task_id] = (
                labels_per_task_dict[task_id][train_indices],
                labels_per_task_dict[task_id][test_indices]
            )

        return (
            train_images_per_task_dict,
            train_labels_per_task_dict,
            test_images_per_task_dict,
            test_labels_per_task_dict
        )

    def get_task_dataset(self, task_id, mode='train'):
        """
        Retrieves a TensorDataset for a specific task and mode ('train' or 'test').

        Parameters:
        - task_id (int): Identifier for the specific task.
        - mode (str): Mode of the dataset, either 'train' or 'test'.

        Returns:
        - dataset (torch.utils.data.TensorDataset): Dataset for the specified task and mode.

        Raises:
        - ValueError: If an invalid mode is specified. Supported values are 'train' or 'test'.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_data = np.load(os.path.join(self.cache_dir, 'tasks', f"{task_id}", f'{mode}.npz'))
        images, labels = task_data["images"], task_data["labels"]

        dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
        dataset.name = f"{task_id}"

        return dataset
