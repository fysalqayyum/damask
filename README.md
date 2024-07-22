# damask  
## The petsc for installation is available at: https://drive.google.com/drive/folders/1Ma9DSO99X-NtWBiE07E26XH8vp5HTA48?usp=sharing  

## Download the DAMASK2.0.2.tar and PETSc.tat files to the Softwares folder  

### Update and Upgrade your Linux Libraries

sudo apt-get update  
sudo apt-get upgrade  

### Install Python

sudo apt-get install python  

### Check and Install dependencies

sudo apt-get install gfortran  
sudo apt-get install cxx  

### To install Petsc copy the downloaded .tar file to the intended installation folder then then doe the following

tar -xf petsc-3.9.3.tar.gz   
cd petsc-3.9.3/  
./configure --with-fc=gfortran --with-cc=gcc --with-cxx=c++ --download-openmpi --download-fftw --download-hdf5 --download-fblaslapack --download-chaco --download-exodusii --download-hypre=https://github.com/hypre-space/hypre/archive/refs/tags/v2.14.0.tar.gz --download-cmake --download-metis --download-ml --download-mumps --download-netcdf --download-parmetis --download-scalapack --download-suitesparse --download-superlu --download-superlu_dist --download-triangle --download-fblaslapack --download-zlib --download-pnetcdf --download-valgrind-devel --with-c2html=0 --with-debugging=0 --with-ssl=0 --with-x=0 COPTFLAGS="-O3 -xHost -no-prec-div" CXXOPTFLAGS="-O3 -xHost -no-prec-div" FOPTFLAGS="-O3 -xHost -no-prec-div"   
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
