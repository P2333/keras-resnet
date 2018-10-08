# jungpu5
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet.py --label_smooth=0.1 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p1_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet.py --label_smooth=0.2 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p2_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet.py --label_smooth=0.3 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet.py --label_smooth=0.8 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p8_True.out 2>&1 &


# jungpu6
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet.py --label_smooth=0.4 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p4_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet.py --label_smooth=0.5 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet.py --label_smooth=0.6 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p6_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet.py --label_smooth=0.7 --augmentation=True > ../cifar10_resnet_Single_labelsmooth0p7_True.out 2>&1 &

