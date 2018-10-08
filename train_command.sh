# jungpu6
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_JSD_test.py --lamda=0 --augmentation=False > ../cifar10_resnet_JSD_lamda0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_JSD_test.py --lamda=0 --augmentation=True > ../cifar10_resnet_JSD_lamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_JSD_test.py --lamda=100.0 --augmentation=False > ../cifar10_resnet_JSD_lamda100p0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_JSD_test.py --lamda=10.0 --augmentation=False > ../cifar10_resnet_JSD_lamda10p0_False.out 2>&1 &


# jungpu5
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_JSD_test.py --lamda=1.0 --augmentation=False > ../cifar10_resnet_JSD_lamda1p0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_JSD_test.py --lamda=1.0 --augmentation=True > ../cifar10_resnet_JSD_lamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_JSD_test.py --lamda=10.0 --augmentation=True > ../cifar10_resnet_JSD_lamda10p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_JSD_test.py --lamda=100.0 --augmentation=True > ../cifar10_resnet_JSD_lamda100p0_True.out 2>&1 &



# jungpu7
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_JSD_test.py --lamda=5.0 --augmentation=False > ../cifar10_resnet_JSD_lamda5p0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_JSD_test.py --lamda=5.0 --augmentation=True > ../cifar10_resnet_JSD_lamda5p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_JSD_test.py --lamda=2.0 --augmentation=False > ../cifar10_resnet_JSD_lamda2p0_False.out 2>&1 &



# jungpu4
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_JSD_test.py --lamda=2.0 --augmentation=True > ../cifar10_resnet_JSD_lamda2p0_True.out 2>&1 &


# jungpu3
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_JSD_test.py --lamda=4.0 --augmentation=False > ../cifar10_resnet_JSD_lamda4p0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_JSD_test.py --lamda=4.0 --augmentation=True > ../cifar10_resnet_JSD_lamda4p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_JSD_test.py --lamda=3.0 --augmentation=False > ../cifar10_resnet_JSD_lamda3p0_False.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_JSD_test.py --lamda=3.0 --augmentation=True > ../cifar10_resnet_JSD_lamda3p0_True.out 2>&1 &













CUDA_VISIBLE_DEVICES=2 python cifar10_resnet.py > ../cifar10_resnet.out 2>&1 &


CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_JSD_test.py --lamda=0
