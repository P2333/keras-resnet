# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda0_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.3 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda0p3_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.7 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda0p7_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=1.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda1p0_leaveone_True.out 2>&1 &


# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_plus_models2_lamda2p0_nonMElamda0_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.3 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_plus_models2_lamda2p0_nonMElamda0p3_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=0.7 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_plus_models2_lamda2p0_nonMElamda0p7_leaveone_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_plus_leaveone.py --lamda=2.0 --nonME_lamda=1.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_plus_models2_lamda2p0_nonMElamda1p0_leaveone_True.out 2>&1 &