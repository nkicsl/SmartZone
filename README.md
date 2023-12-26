
# SmartZone

SmartZone is a secure memory management method based on TrustZone and includes the lightweight neural network framework Tinylib. This application is based on [Darknet DNN framework](https://pjreddie.com/darknet/) and needs to be run with [OP-TEE](https://www.op-tee.org/), an open source framework for Arm TrustZone.

# Description
This is the code repository for our paper "Memory-Efficient and Secure DNN Inference on TrustZone-enabled Consumer IoT Devices". (paper id: 1570937997)

# Environment
For execution
- Development board: Raspberry pi 3B+ (ARM Cortex A53 * 4, 1GB RAM)
- REE OS: Linux-for-arm 4.14
- TEE OS: OP-TEE 3.8.0
- Simulator: QEMU v8

For compilation 
- Host OS for compiling OP-TEE: Ubuntu-22.04, 64bit, x86_64
- GCC version：11.3.0
- toolchain: aarch64-linux-gnu-gcc 10.2.1


# Setup
## (1) Set up OP-TEE
1) Follow **step1** ~ **step5** in "**Get and build the solution**" to build the OP-TEE solution.
https://optee.readthedocs.io/en/latest/building/gits/build.html#get-and-build-the-solution

2) **For real boards**: If you are using boards, keep follow **step6** ~ **step7** in the above link to flash the devices. This step is device-specific.

   **For simulation**: If you have chosen QEMU-v7/v8, run the below command to start QEMU console.
```
make run
(qemu)c
```

3) Follow **step8** ~ **step9** to test whether OP-TEE works or not. Run:
```
tee-supplicant -d
xtest
```

Note: you may face OP-TEE related problem/errors during setup, please also free feel to raise issues in [their pages](https://github.com/OP-TEE/optee_os).

## (2) Build Tinylib
1) clone codes and datasets
```
git clone https://github.com/nkicsl/SmartZone.git
```
Let `$PATH_OPTEE$` be the path of OPTEE, `$PATH_Tinylib$` be the path of Tinylib, and `$PATH_datasets$` be the path of dataset.

2) copy Tinylib to example dir
```
mkdir $PATH_OPTEE$/optee_examples/Tinylib
cp -a $PATH_Tinylib$/. $PATH_OPTEE$/optee_examples/Tinylib/
```

3) copy datasets to root dir
```
cp -a $PATH_datasets$/. $PATH_OPTEE$/out-br/target/root/
```

4) rebuild the OP-TEE

**For simulation**, to run `make run` again.

**For real boards**, to run `make flash` to flash the OP-TEE with `darknetz` to your device.



# Inference

By simply typing the following command, you can do inference using a pre-trained model.
```
tinylib classifier predict cfg/imagenet1k.data cfg/mobilenet_v1.cfg models/mobilenet_v1.weights data/cat.jpg -mov_size 262144
```
You can adjust the parameter -mov_size to change the size of shared memory

### File description
- ``Compiled file/``: Compiled binary files
- ``Tinylib/``: N-Tinylib,S-Tinylib,Tinylibm
- ``dataset/``: the experimental data of our paper

