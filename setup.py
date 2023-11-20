from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    install_requires = [line.strip() for line in requirements_file]

SHORT_DESCRIPTION = (
    "FedKit-Learn is a Python package facilitating federated machine learning simulations,"
    " attack scenarios, and dataset management for streamlined research and development in"
    " decentralized machine learning."
)

setup(
    name='fedklearn',
    version='0.1.0',
    author='Othmane Marfoq',
    author_email='othmane.marfoq@inria.fr',
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/omarfoq/fedkit-learn',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(),
    install_requires=install_requires,
)
