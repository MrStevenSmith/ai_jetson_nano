#!/usr/bin/env bash

# https://github.com/MrStevenSmith/ai_jetson_nano
# Steven Smith
# v1.0 - 12 March 2020

# Read sudo password
clear
echo " "
echo "Jetson Nano Initial Configuration Script"
echo "========================================"
echo " "
echo "Run this script to set up your Jetson Nano.  Only run the script once."
echo "Before running a second time re-image your Jetson Nano with the Nvidia Jetpack Image"
echo " "
echo " "
echo "Please type in the sudo password"
read -s PASSWORD
echo " "

# Set NTP servers
echo " "
echo "1. Setting NTP servers"
echo " "
sleep 6
cd ~
FILE="/etc/systemd/timesyncd.conf"
echo $PASSWORD | sudo -S bash -c "echo 'NTP=0.arch.pool.ntp.org 1.arch.pool.ntp.org 2.arch.pool.ntp.org 3.arch.pool.ntp.org' >> $FILE"
echo $PASSWORD | sudo -S bash -c "echo 'FallbackNTP=0.pool.ntp.org 1.pool.ntp.org 0.us.pool.ntp.org' >> $FILE"
echo $PASSWORD | sudo -S systemctl restart systemd-timesyncd.service

# Make 6G swap file
echo " "
echo "2. Making 6G swap file"
echo " "
sleep 6
echo $PASSWORD | sudo -S fallocate -l 6G /var/swapfile
echo $PASSWORD | sudo -S chmod 600 /var/swapfile
echo $PASSWORD | sudo -S mkswap /var/swapfile
echo $PASSWORD | sudo -S swapon /var/swapfile
echo $PASSWORD | sudo -S bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

# Create ai root
mkdir ~/ai

# Disable syslog to prevent large log files from collecting
echo " "
echo "3. Disabling syslog"
echo " "
sleep 6
echo $PASSWORD | sudo -S service rsyslog stop
echo $PASSWORD | sudo -S systemctl disable rsyslog

# Update installed packages
echo " "
echo "4. Updating installed packages"
echo " "
sleep 6
echo $PASSWORD | sudo -S apt install apt-utils -y
echo $PASSWORD | sudo -S apt update
echo $PASSWORD | sudo -S apt --yes --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confnew" upgrade
echo $PASSWORD | sudo -S apt autoremove -y

# Install jupyter lab

## Install dependancies
echo " "
echo "5. Installing dependancies"
echo " "
sleep 6
echo $PASSWORD | sudo -S apt install nano unzip libfreetype6-dev libpng-dev libatlas-base-dev libopenblas-base libopenblas-dev cmake liblapack-dev libjpeg-dev gfortran python3-opencv pkg-config libhdf5-100 libhdf5-dev -y

## Install pip3
echo $PASSWORD | sudo -S apt install python3-pip -y

## Install virtualenv
echo $PASSWORD | sudo -S apt install python3-venv -y

## Create ai virtual environment
echo " "
echo "6. Creating Python virtual environment"
echo " "
python3 -m venv ~/python-envs/ai

## Activate virtual environment
source /home/$USER/python-envs/ai/bin/activate

## Install wheel
echo " "
echo "7. Installing further dependancies"
echo " "
pip3 install wheel

## Install cython
pip3 install cython

## Install pybind11
pip3 install pybind11

## Install jupyter lab
echo " "
echo "8. Installing Jupyter Lab"
echo " "
sleep 6
pip3 install jupyter jupyterlab

## Create Jupyter configuration file
~/python-envs/ai/bin/jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo c.NotebookApp.port = 8888 >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.password = u'sha1:43819219905f:8399d9512c2df00376e1107bd3e3352cf18fc366'" >> ~/.jupyter/jupyter_notebook_config.py
echo c.NotebookApp.open_browser = False >> ~/.jupyter/jupyter_notebook_config.py

## create env file in ~/.jupyter/env
touch ~/.jupyter/env
echo PATH=~/python-envs/ai/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin:$PATH >> ~/.jupyter/env
echo LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH >> ~/.jupyter/env
echo CUDA_HOME=/usr/local/cuda >> ~/.jupyter/env
echo CUDA_VISIBLE_DEVICES=0 >> ~/.jupyter/env

## Enable i2c permissions
echo $PASSWORD | sudo -S usermod -aG i2c $USER

## Create jupyter lab service
echo $PASSWORD | sudo -S cat << EOF > /tmp/jupyter.service
# service name:     jupyter.service 
# path:             /etc/systemd/jupyter.service

[Unit]
Description=Jupyter Lab Service

[Service]
Type=simple
PIDFile=/run/jupyter.pid
EnvironmentFile=/home/$USER/.jupyter/env
ExecStart=/home/$USER/python-envs/ai/bin/jupyter-lab --config=/home/$USER/.jupyter/jupyter_notebook_config.py --no-browser
User=$USER
Group=$USER
WorkingDirectory=/home/$USER/ai
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
echo $PASSWORD | sudo -S mv /tmp/jupyter.service /etc/systemd/system/jupyter.service
echo $PASSWORD | sudo -S systemctl enable jupyter
echo $PASSWORD | sudo -S systemctl daemon-reload
echo $PASSWORD | sudo -S systemctl restart jupyter

## Install pima dependancies
echo " "
echo "9. Installing Pima Indian dependancies"
echo " "
sleep 6
pip3 install "pillow<7"
pip3 install ipython matplotlib numpy 
pip3 install pandas rise scipy
pip3 install scikit-learn seaborn
mv ~/initial_setup/datasets ~/ai
mv ~/initial_setup/notebooks ~/ai

# Install DoorCam

## Install doorcam dependancies
echo " "
echo "10. Installing DoorCam dependancies"
echo " "
sleep 6
pip3 install dlib
pip3 install face_recognition

## Download doorcam script
echo " "
echo "11. Installing DoorCam"
echo " "
sleep 6
wget -O doorcam.py tiny.cc/doorcam
mkdir ~/ai/doorcam
mv doorcam.py ~/ai/doorcam/

# Install 2 days to demo

## Install PyTorch 1.2.0
echo " "
echo "12. Installing PyTorch v1.2.0"
echo " "
sleep 6
wget https://nvidia.box.com/shared/static/06vlvedmqpqstu1dym49fo7aapgfyyu9.whl -O torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl
rm torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl

## Install Torchvision 0.4.0
echo " "
echo "13. Installing Torchvision v0.4.x"
echo " "
sleep 6
git clone --branch v0.4.0 https://github.com/pytorch/vision /home/$USER/ai/torchvision
cd /home/$USER/ai/torchvision
python3 setup.py install
cd /home/$USER

## Install TensorFlow
echo " "
echo "14. Installing TensorFlow and Models"
echo " "
sleep 6
pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu

#### Install TensorFlow models repository
echo " "
echo "15. Installing TensorFlow Models"
echo " "
cd /home/$USER/ai
git clone https://github.com/tensorflow/models
cd models/research
git checkout 5f4d34fc
wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-aarch_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
python3 setup.py install
cd slim
python3 setup.py install
cd /home/$USER

## Install 2 days to demo
echo " "
echo "16. Installing 2 Days to a Demo"
echo " "
sleep 6
deactivate
echo $PASSWORD | sudo -S apt-get install -y dialog
echo $PASSWORD | sudo -S apt-get install -y libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
echo $PASSWORD | sudo -S apt-get install -y libopencv-calib3d-dev libopencv-dev qtbase5-dev qt5-default doxygen
# libgstreamer0.10-0-dev libgstreamer-plugins-base0.10-dev libxml2-dev
cd /home/$USER/ai
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
cat << EOF > ~/ai/jetson-inference/CMakePreBuild.sh
#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD

PASSWORD=$1

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:   $BUILD_ROOT"
echo " "


# break on errors
#set -e


# run the model downloader
#./download-models.sh


# run the pytorch installer
#./install-pytorch.sh


echo "[Pre-build]  Finished CMakePreBuild script"
EOF
mkdir build
cd build
echo $PASSWORD | sudo -S cmake ../
echo $PASSWORD | sudo -S make
echo $PASSWORD | sudo -S make install
echo $PASSWORD | sudo -S ldconfig
cd /home/$USER
source /home/$USER/python-envs/ai/bin/activate

## Install models
echo " "
echo "17. Installing 2 Days to a Demo Models"
echo " "
cd /home/$USER/ai/jetson-inference/data/networks
wget https://nvidia.box.com/shared/static/at8b1105ww1c5h7p30j5ko8qfnxrs0eg.caffemodel -O ~/ai/jetson-inference/data/networks/bvlc_googlenet.caffemodel
wget https://nvidia.box.com/shared/static/5z3l76p8ap4n0o6rk7lyasdog9f14gc7.prototxt -O ~/ai/jetson-inference/data/networks/googlenet.prototxt
wget https://nvidia.box.com/shared/static/ue8qrqtglu36andbvobvaaj8egxjaoli.prototxt -O ~/ai/jetson-inference/data/networks/googlenet_noprob.prototxt
wget https://nvidia.box.com/shared/static/gph1qfor89vh498op8cicvwc13zltu3h.gz ~/ai/jetson-inference/data/networks
tar xf gph1qfor89vh498op8cicvwc13zltu3h.gz
rm gph1qfor89vh498op8cicvwc13zltu3h.gz
wget https://nvidia.box.com/shared/static/jcdewxep8vamzm71zajcovza938lygre.gz ~/ai/jetson-inference/data/networks
tar xf jcdewxep8vamzm71zajcovza938lygre.gz
rm jcdewxep8vamzm71zajcovza938lygre.gz
wget https://nvidia.box.com/shared/static/0wbxo6lmxfamm1dk90l8uewmmbpbcffb.gz ~/ai/jetson-inference/data/networks
tar xf 0wbxo6lmxfamm1dk90l8uewmmbpbcffb.gz
rm 0wbxo6lmxfamm1dk90l8uewmmbpbcffb.gz
wget https://nvidia.box.com/shared/static/wjitc00ef8j6shjilffibm6r2xxcpigz.gz ~/ai/jetson-inference/data/networks
tar xf wjitc00ef8j6shjilffibm6r2xxcpigz.gz
rm wjitc00ef8j6shjilffibm6r2xxcpigz.gz
wget https://nvidia.box.com/shared/static/3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz ~/ai/jetson-inference/data/networks
tar xf 3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz
rm 3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz
wget https://nvidia.box.com/shared/static/k7s7gdgi098309fndm2xbssj553vf71s.gz ~/ai/jetson-inference/data/networks
tar xf k7s7gdgi098309fndm2xbssj553vf71s.gz
rm k7s7gdgi098309fndm2xbssj553vf71s.gz
wget https://nvidia.box.com/shared/static/9aqg4gpjmk7ipz4z0raa5mvs35om6emy.gz ~/ai/jetson-inference/data/networks
tar xf 9aqg4gpjmk7ipz4z0raa5mvs35om6emy.gz
rm 9aqg4gpjmk7ipz4z0raa5mvs35om6emy.gz
wget https://nvidia.box.com/shared/static/jm0zlezvweiimpzluohg6453s0u0nvcv.gz ~/ai/jetson-inference/data/networks
tar xf jm0zlezvweiimpzluohg6453s0u0nvcv.gz
rm jm0zlezvweiimpzluohg6453s0u0nvcv.gz
wget https://nvidia.box.com/shared/static/dgaw0ave3bdws1t5ed333ftx5dbpt9zv.gz ~/ai/jetson-inference/data/networks
tar xf dgaw0ave3bdws1t5ed333ftx5dbpt9zv.gz
rm dgaw0ave3bdws1t5ed333ftx5dbpt9zv.gz
wget https://nvidia.box.com/shared/static/p63pgrr6tm33tn23913gq6qvaiarydaj.gz ~/ai/jetson-inference/data/networks
tar xf p63pgrr6tm33tn23913gq6qvaiarydaj.gz
rm p63pgrr6tm33tn23913gq6qvaiarydaj.gz
wget https://nvidia.box.com/shared/static/5vs9t2wah5axav11k8o3l9skb7yy3xgd.gz ~/ai/jetson-inference/data/networks
tar xf 5vs9t2wah5axav11k8o3l9skb7yy3xgd.gz
rm 5vs9t2wah5axav11k8o3l9skb7yy3xgd.gz

## Download cat_dog dataset
echo " "
echo "18. Downloading cat_dog dataset"
echo " "
sleep 6
wget https://nvidia.box.com/shared/static/o577zd8yp3lmxf5zhm38svrbrv45am3y.gz -O ~/ai/datasets/cat_dog.tar.gz
cd /home/$USER/ai/datasets
tar xzf cat_dog.tar.gz
rm cat_dog.tar.gz
cd /home/$USER

## Download PlantCLEF dataset
echo " "
echo "19. Downloading PlantCLEF dataset"
echo " "
sleep 6
wget https://nvidia.box.com/shared/static/vbsywpw5iqy7r38j78xs0ctalg7jrg79.gz -O ~/ai/datasets/PlantCLEF_Subset.tar.gz
cd /home/$USER/ai/datasets
tar xzf PlantCLEF_Subset.tar.gz
rm PlantCLEF_Subset.tar.gz
cd /home/$USER

# Removed un-used packages
echo " "
echo "20. Removing un-used packages"
echo " "
sleep 6
echo " "
echo $PASSWORD | sudo -S apt autoremove -y

# Output versions
#clear
echo " "
echo "21. Outputting installed AI packages"
echo " "
echo " "
sleep 6
echo "TensorFlow Version"
python3 -c 'import tensorflow; print(tensorflow.__version__)'
echo " "
echo "PyTorch Version"
python3 -c 'import torch; print(torch.__version__)'
echo " "
echo "Torchvision Version"
python3 -c 'import torchvision; print(torchvision.__version__)'
echo " "
echo "Jupyter Version"
python3 -c 'import jupyter; print(jupyter.__version__)'
echo " "
echo " "
echo "Tools have been installed in a Virtual Pthon Environment"
echo " "
echo "To enter the Virtual Environment type 'source ~/python-envs/ai/bin/activate'"
echo " "
echo "To exit the Virtual Environment type 'deactivate'"
echo " "
echo " "
echo "To install new models, use the following commands: -"
echo " "
echo "    $ cd <jetson-inference>/tools"
echo "    $ ./download-models.sh"

# Reboot
echo " "
echo " "
echo " "
echo "System will reboot in 20 seconds."
echo " "
rm -rf ~/initial_setup
sleep 20
echo $PASSWORD | sudo -S reboot
