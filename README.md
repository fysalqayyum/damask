# damask  

## You should have Ubuntu 20.04.6 LTS version installed before starting

## The petsc for installation files are available at: https://drive.google.com/drive/folders/1Ma9DSO99X-NtWBiE07E26XH8vp5HTA48?usp=sharing  

## Download the DAMASK2.0.2.tar and PETSc.tat files to the Softwares folder  

### Update and Upgrade your Linux Libraries

sudo apt-get update  
  
sudo apt-get upgrade  

### Install Python

sudo apt-get install python-is-python2  

### Check and Install dependencies

sudo apt-get install gfortran  
  
sudo apt-get install cmake gcc g++

sudo apt-get install m4

### To install Petsc copy the downloaded .tar file to the intended installation folder then then doe the following

tar -xf petsc-3.9.3.tar.gz   
  
cd petsc-3.9.3/  
  
./configure --with-fc=gfortran --with-cc=gcc --with-cxx=c++ --download-openmpi --download-fftw --download-hdf5 --download-fblaslapack --download-chaco --download-exodusii --download-hypre=https://github.com/hypre-space/hypre/archive/refs/tags/v2.14.0.tar.gz --with-cmake --download-metis --download-ml --download-mumps --download-netcdf=https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.5.0.tar.gz --download-parmetis --download-make --download-scalapack --download-suitesparse --download-superlu --download-superlu_dist --download-triangle --download-fblaslapack --download-zlib --download-pnetcdf --download-valgrind-devel --with-c2html=0 --with-debugging=0 --with-ssl=0 --with-x=0 COPTFLAGS="-O3 -xHost -no-prec-div" CXXOPTFLAGS="-O3 -xHost -no-prec-div" FOPTFLAGS="-O3 -xHost -no-prec-div"   
  
make PETSC_DIR=/home/(your own link to the petsec installation directory)/petsc-3.9.3 PETSC_ARCH=arch-linux2-c-opt all   
  
make PETSC_DIR=/home/(your own link to the petsec installation directory)/petsc-3.9.3 PETSC_ARCH=arch-linux2-c-opt check  

### After Petsc is sucessfully configured, to install DAMASK copy the downloaded .tar file to the intended installation folder then then doe the following

tar -xf DAMASK-2.0.2.tar.xz   
  
cd DAMASK/  
  
export  PETSC_DIR=/home/(your own link to the petsec installation directory)/petsc-3.9.3 PETSC_ARCH=arch-linux2-c-opt   
  
source DAMASK_env.sh   
  
sudo apt-get install python-numpy  
  
sudo apt-get install cmake  
  
make spectral  
  
make processing  
  
DAMASK_spectral

## To install DAMASK everytime you open up a new termial

nano ~/.bashrc

## At the end of this file type by carefully checking for your own directories and addresses

export PETSC_DIR=/home/(your own link to the petsec installation directory)/petsc-3.9.3 PETSC_ARCH=arch-linux2-c-opt

source ~/Software/DAMASK/DAMASK_env.sh

source make ~/Software/DAMASK/processing

# Install further Dependncies for posr processing

sudo apt-get install curl

curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py

sudo python2 get-pip.py

pip install vtk

pip install scipy
