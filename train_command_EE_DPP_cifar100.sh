# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p3_True.out 2>&1 &
