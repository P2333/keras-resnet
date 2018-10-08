# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_plus.py --lamda=0.0 --nonME_lamda=0.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_plus_models2_lamda0_nonMElamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=1.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.3 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_plus_models3_lamda2p0_nonMElamda0p3_True.out 2>&1 &


# jungpu5
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.0 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_plus_models4_lamda2p0_nonMElamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.5 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_plus_models4_lamda2p0_nonMElamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=1.0 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_plus_models4_lamda2p0_nonMElamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.3 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_plus_models4_lamda2p0_nonMElamda0p3_True.out 2>&1 &


# jungpu6
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.0 --num_models=5 --augmentation=True > ../cifar10_resnet_EE_plus_models5_lamda2p0_nonMElamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.5 --num_models=5 --augmentation=True > ../cifar10_resnet_EE_plus_models5_lamda2p0_nonMElamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=1.0 --num_models=5 --augmentation=True > ../cifar10_resnet_EE_plus_models5_lamda2p0_nonMElamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_plus.py --lamda=2.0 --nonME_lamda=0.3 --num_models=5 --augmentation=True > ../cifar10_resnet_EE_plus_models5_lamda2p0_nonMElamda0p3_True.out 2>&1 &


















# Eval
CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_plus-eval.py --lamda=0.0 --nonME_lamda=0.0 --num_models=2 --augmentation=True --epoch=106
CUDA_VISIBLE_DEVICES=1 python cifar10_resnet_EE_plus-eval.py --lamda=2.0 --nonME_lamda=0.5 --num_models=2 --augmentation=True --epoch=150
CUDA_VISIBLE_DEVICES=2 python cifar10_resnet_EE_plus-eval.py --lamda=2.0 --nonME_lamda=0.9 --num_models=2 --augmentation=True --epoch=160

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_plus-eval.py --lamda=2.0 --nonME_lamda=0.0 --num_models=3 --augmentation=True --epoch=187
CUDA_VISIBLE_DEVICES=1 python cifar10_resnet_EE_plus-eval.py --lamda=2.0 --nonME_lamda=0.5 --num_models=3 --augmentation=True --epoch=155
CUDA_VISIBLE_DEVICES=2 python cifar10_resnet_EE_plus-eval.py --lamda=2.0 --nonME_lamda=1.0 --num_models=3 --augmentation=True --epoch=180


CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_plus-eval.py --lamda=0.0 --nonME_lamda=0.0 --num_models=2 --augmentation=True --epoch=143













#Test
CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_plus-adveval.py --lamda=2.0 --nonME_lamda=0.3 --num_models=2 --augmentation=True --epoch=183

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_plus-adveval.py --lamda=2.0 --nonME_lamda=0.3 --num_models=3 --augmentation=True --epoch=176
