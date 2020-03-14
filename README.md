# Jetson Nano Setup Script

Script to install and configure a Jetson Nano to get started with AI


## Getting Started

These instructions will will take the base Jetson Nano image and install the pre-requisites required to cary out the [Getting Started with AI on Jetson Nano](https://courses.nvidia.com/courses/course-v1:DLI+C-RX-02+V1/about) deep learning demonstrations and the [Hello AI World / 2 Days to a Demo (DIGITS)](https://github.com/dusty-nv/jetson-inference) deep learning demonstrations.

### Prerequisites

Before running this script you need to image the MicroSD card with the Jetson Nano JetPack Image.  This script was built and tested using [Jetpack 4.3 image r3231](https://developer.nvidia.com/jetson-nano-sd-card-image-r3231).  The latest JetPack image can be found at the [Jetson Download Center](https://developer.nvidia.com/embedded/downloads).

```
1. Download and run SD Memory Card Formatter
(https://www.sdcard.org/downloads/formatter/)

2. Format the MicroSD card

3. Download JetPack 4.3
(https://developer.nvidia.com/jetson-nano-sd-card-image-r3231)

4. Download and run balenaEtcher
(https://www.balena.io/etcher)

5. Image the MicroSD card using the JetPack 4.3 image

6. Boot the Jetson Nano to the JetPack 4.3 image and carry out the initial configuration
(Accept EULA, select language, select keyboard, select timezone and create username and password)
```

### Installing

Once booted into the JetPack linux build open a terminal and carry out the steps below.

```
cd ~

git clone https://github.com/MrStevenSmith/ai_jetson_nano.git ~/initial_setup

chmod +x ~/initial_setup/initial_setup.sh

~/initial_setup/initial_setup.sh
```

The script can take over an hour to complete, depending on the speed of your internet connection.  At the beginning of the script you will need to enter the sudo password.

### Installing New Jetson Inference Models

To install new models run the following script.

```
cd ~/ai/jetson-inference/tools

./download-models.sh
```

### TensorFlow Models

TensorFlow Models can be found in ~/ai/tensorflow-models


## Running The Deep Learning Demonstrations

Once the script has finished all pre-requisites should be installed and configured to carry out the below deep learning demonstrations.

### Getting Started with AI on Jetson Nano

Start the demonstrations here: -

```
https://courses.nvidia.com/courses/course-v1:DLI+C-RX-02+V1/courseware/b2e02e999d9247eb8e33e893ca052206/63a4dee75f2e4624afbc33bce7811a9b/?activate_block_id=block-v1%3ADLI%2BC-RX-02%2BV1%2Btype%40sequential%2Bblock%4063a4dee75f2e4624afbc33bce7811a9b
```

### Hello AI World

Start the demonstrations here: -

```
https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md
```

### 2 Days to a Demo (DIGITS)

Start the demonstrations here: -

```
https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console.md
```

When running the Hello AI World and 2 Days to a Demo labs any commands starting wih 'python' need to be run using 'python3' instead.

### Pima Indian Diabetes Notebook

The Pima Indians Diabetes demo can be run from Jupyter Notebook.  The Notebook is saved in the Notebooks folder.


## Python Virtual Environment

All python packages have been install inside a virtual environment.  To use the virtual environment run the following command: -

```
source ~/python-env/ai/bin/activate
```


## Jupyter Notebook Details

The initial password for Jupyter Notebook is: -

```
Jupyter
```

To access Jupyter Notebook remotely go to the following url: -

```
http://<IP Address of Jetson Nano>:8888/
```

To change the Jupyter password you must carry out the below.

```
1. Open the file ~/.jupyter/jupyter_notebook_config.py and at the bottom of the file # out the line starting: -
'c.NotebookApp.password = u'

2. Run the command 'sudo -S jupyter notebook password'
```


## Built With

* Python3
* PyTorch 1.2.0
* Torchvision 0.4.0
* TensorFlow
* Jupyter
* Python Virtual Environment


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details


## Author

* **Steven Smith** - *Create the ansible playbook* - [MrStevenSmith](https://github.com/MrStevenSmith)

### Acknowledgments
* **Dustin Franklin** - *Created jetson-inference code* - [dusty-nv](https://github.com/dusty-nv)
* **Adam Geitgey** - *Created the doorcam demo* - [ageitgey](https://gist.github.com/ageitgey)
* **Andrea Grandi** - *Created the Pima Indians Diabetes Jupyter Notebook* - [andreagrandi](https://github.com/andreagrandi)