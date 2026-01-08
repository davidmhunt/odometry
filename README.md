# Odometry
Odometry package for estimating the state of agent's using various sensors

## Installation
In order for the code to work properly, the following steps are required
1. Install correct version of python
2. Install CPSL_TI_Radar using Poetry

### 1. Setup Python environment

#### Deadsnakes PPA (requires sudo access)
1. On ubuntu systems, start by adding the deadsnakes PPA to add the required version of python.
```
sudo add-apt-repository ppa:deadsnakes/ppa
```

2. Update the package list
```
sudo apt update
```

3. Install python 3.12 along with the required development dependencies
```
sudo apt install python3.12 python3.12-dev
```

The following resources may be helpful [Deadsnakes PPA description](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa), [Tutorial on Deadsnakes on Ubuntu](https://preocts.github.io/python/20221230-deadsnakes/)

#### Conda (Backup)
1. If conda isn't already installed, follow the [Conda Install Instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to install conda
2. Use the following command to download the conda installation (for linux)
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Run the conda installation script (-b for auto accepting the license)
```
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
```
3. Once conda is installed, create a new conda environment with the correct version of python
```
conda create -n odometry python=3.12
```

### 2. Clone CPSL_TI_Radar
```
git clone https://github.com/davidmhunt/Odometry.git
```
Initialize the submodules
```
cd odometry
git submodule update --init
```
### 3. Install CPSL_TI_Radar using Poetry

#### Installing Poetry:
 
1. Check to see if Python Poetry is installed. If the below command is successful, poetry is installed move on to setting up the conda environment

```
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

If you are using poetry over an ssh connection or get an error in the following steps, try running the following command first and then continuing with the remainder fo the installation.
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
### Installing odometry (with torch)
If your machine supports it Navigate to the odometry foler (this folder) and execute the following command

```
poetry install --extras "submodules"
```



### Installing odometry (if torch fails to install)
If your machine supports it Navigate to the odometry foler (this folder) and execute the following command

```
poetry install --with submodules
```

Follow the following instructions to install the correct version of pytorch for your system.

1. Navigate to the [pytorch installation page](https://pytorch.org/get-started/locally/). Select the requirements for your system. However, under the "package" select the "Pip" option. Once you have specified the options for your system, you'll get a command similar to this
```
pip3 install torch torchvision torchaudio torchsummary torch_geometric
```
2. Navigate to the odometry folder
```
cd odometry
```
3. Start a poetry shell
```
eval $(poetry env activate)
```
4. run the command given by the pytorch website
```
pip3 install torch torchvision torchaudio
```
5. If this runs normally, you should now be good to exit the poetry shell
```
exit
```

### Installing torch-cluster
Currently, torch-cluster is required to run everything. At the current moment though, this cannot be installed using poetry. To overcome this, run the following commands to correctly install everything. Here, replace ${CUDA} with cpu, cu118, cu121, or cu124 depending on cuda version. While this command should work for most systems, see the following page for more specific instructions: [torch-cluster github](https://github.com/rusty1s/pytorch_cluster) 
```
cd odometry
eval $(poetry env activate)
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+${CUDA}.html
```

#### Updating Odometry
If the pyproject.toml file is updated, the poetry installation must also be updated. Use the following commands to update the version of poetry
```
poetry lock --no-update
poetry install
```

### Using .env for Project Directories

In order to use any datasets in your computer's directory, you must first create a .env file and mark where the dataset files can be found.

1. Create a .env file in your project's root directory. This will file will not be uploaded to GitHub when you commit your changes.
2. Inside the .env file, add these variables
```
DATASET_DIRECTORY=/data/radnav/rad_nav_datasets/
MODEL_DATASET_DIRECTORY=/data/radnav/
MAP_DIRECTORY=/data/radnav/maps/
CONFIG_DIRECTORY=/home/USERNAME/Documents/odometry/submodules/mmwave_radar_processing/configs/
MOVIE_TEMP_DIRECTORY=/home/USERNAME/Downloads/radnav_temp_dir
RADCLOUD_MODEL_STATE_DICT_PATH=/home/USERNAME/Documents/odometry/submodules/radcloud/working_dir/RadCloud_40_chirps_10e.pth
RADARHD_MODEL_STATE_DICT_PATH=/data/RadarHD/radarhd_dict.pt_gen
```
3. Replace the example text with the path to your directory

## Loading the radarHD model

```
https://drive.google.com/file/d/1JorZEkDCIcQDSaMAabvkQX4scvwj0wzn/view?usp=sharing
```