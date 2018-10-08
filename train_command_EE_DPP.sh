# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_DPP_models2_lamda2p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.3 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_DPP_models2_lamda2p0_logdetlamda0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.7 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_DPP_models2_lamda2p0_logdetlamda0p7_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=1.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_DPP_models2_lamda2p0_logdetlamda1p0_True.out 2>&1 &


# jungpu7
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.7 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda0p7_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=1.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda1p0_True.out 2>&1 &


# jungpu6
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.0 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_DPP_models4_lamda2p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.3 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_DPP_models4_lamda2p0_logdetlamda0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.7 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_DPP_models4_lamda2p0_logdetlamda0p7_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=1.0 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_DPP_models4_lamda2p0_logdetlamda1p0_True.out 2>&1 &
