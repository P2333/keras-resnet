# jungpu9
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE.py --lamda=0 --augmentation=True > ../cifar10_resnet_EE_lamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u cifar10_resnet_EE.py --lamda=1.0 --augmentation=True > ../cifar10_resnet_EE_lamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE.py --lamda=2.0 --augmentation=True > ../cifar10_resnet_EE_lamda2p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python -u cifar10_resnet_EE.py --lamda=3.0 --augmentation=True > ../cifar10_resnet_EE_lamda3p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u cifar10_resnet_EE.py --lamda=5.0 --augmentation=True > ../cifar10_resnet_EE_lamda5p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u cifar10_resnet_EE.py --lamda=10.0 --augmentation=True > ../cifar10_resnet_EE_lamda10p0_True.out 2>&1 &
