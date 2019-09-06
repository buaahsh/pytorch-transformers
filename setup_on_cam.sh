echo "----------------- Check Env -----------------"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V
python -V
python -c 'import torch; print(torch.__version__)'
echo "PHILLY_GPU_COUNT" ${PHILLY_GPU_COUNT}

echo "----------------- Check File System -----------------"
echo "I am " $(whoami)
echo -n "CURRENT_DIRECTORY "
pwd
echo "PHILLY_HOME" ${PHILLY_HOME}
ls -alh ${PHILLY_HOME}
echo "PHILLY_USER_DIRECTORY" ${PHILLY_USER_DIRECTORY}
ls -alh ${PHILLY_USER_DIRECTORY}

echo "----------------- Start Installing -----------------"
PWD_DIR=$(pwd)
cd $PHILLY_SCRATCH_DIRECTORY
cd $(mktemp -d)
echo "----------------- Install NCCL -----------------"
cp /mnt/phil/pkg/nccl-ubuntu1604-2.4.2-cuda10.0.deb .
sudo dpkg -i nccl-ubuntu1604-2.4.2-cuda10.0.deb
sudo apt update
sudo apt install libnccl2=2.4.2-1+cuda10.0 libnccl-dev=2.4.2-1+cuda10.0
# echo "----------------- Install Apex -----------------"
# git clone -q https://github.com/NVIDIA/apex.git
# cd apex
# git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
# python setup.py install --user --cuda_ext --cpp_ext
echo "----------------- Install Python Package -----------------"
# pip install --user allennlp
pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn
cd $PWD_DIR
echo "----------------- Start Training -----------------"
# cd $PHILLY_SCRATCH_DIRECTORY
# mkdir src
# cd src
# cp -r $PT_CODE_DIR/* .
pip install --user --editable .
