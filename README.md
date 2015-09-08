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

### Running on an Amazon EC2 Instance with GPU

Run a EC2 GPU-optimized instance, for example `g2.2xlarge`. 

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev  gfortran libblas-dev liblapack-dev libatlas-base-dev
pip install -U pip
pip install virtualenv

# Install CUDA 7
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update && sudo apt-get -y install cuda

# Clone the repo and install requirements
git clone git@github.com:dennybritz/nn-theano.git
cd nn-theano
virtualenv venv && source venv/bin/activate
pip install -r requirements.txt

# Environment variables
export CUDA_ROOT=/usr/local/cuda
```

Then, follow the [jupyter instructions to run a public notebook server](http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#notebook-public-server).