# Optimizing Neural Network implementation with Theano

### Local Jupyter notebook setup

```bash
# Create a new virtual environment (optional)
virtualenv venv
# Install requirements
pip install -r requirements.txt
# Start the notebook server
jupyter notebook .
```

### Running on an GPU-optimized Amazon EC2 instance

Run a EC2 GPU-optimized instance, for example `g2.2xlarge`. 

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev  gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-image-generic
sudo pip install -U pip

# Install CUDA 7
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
sudo reboot

# Clone the repo and install requirements
git clone git@github.com:dennybritz/nn-theano.git
cd nn-theano
sudo pip install -r requirements.txt

# Set Environment variables
export CUDA_ROOT=/usr/local/cuda-7.0
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
```

Then, follow the [jupyter instructions to run a public notebook server](http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#notebook-public-server).