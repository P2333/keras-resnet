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

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=1.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda2p0_logdetlamda1p0_True.out 2>&1 &


# jungpu6
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=2 --augmentation=True > ../cifar10_resnet_EE_DPP_models2_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=4 --augmentation=True > ../cifar10_resnet_EE_DPP_models4_lamda0_logdetlamda0_True.out 2>&1 &


#jungpu5
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda1p0_logdetlamda0_True.out 2>&1 &




#Plot figure
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.4 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda1p4_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.4 --log_det_lamda=0.3 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda1p4_logdetlamda0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_models3_lamda0_logdetlamda0p3_True.out 2>&1 &



# Same as original paper's training setting on CIFAR-10, totally 160epoch
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_SameAsPaper_models3_lamda1p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_SameAsPaper_models3_lamda2p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_SameAsPaper_models3_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar10_resnet_EE_DPP_SameAsPaper_models3_lamda1p0_logdetlamda0_True.out 2>&1 &









# Eval
CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.0 --num_models=2 --augmentation=True --epoch=139

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=2 --augmentation=True --epoch=153

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.7 --num_models=2 --augmentation=True --epoch=194

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=4 --augmentation=True --epoch=170

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch=180

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=2.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch=157

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch=139




# Nor acc eval
CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval_nor_acc.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-eval_nor_acc.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch=139






#Adv ensemble acc (noniterative)
CUDA_VISIBLE_DEVICES=6 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --attack_method=SaliencyMapMethod > adv_ensemble_acc_SaliencyMapMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --attack_method=DeepFool > adv_ensemble_acc_DeepFool_model3_lamda2p0_loglamda0p5.out 2>&1 &

#Adv ensemble acc (target)


#Adv ensemble acc
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MadryEtAl > adv_ensemble_acc_MadryEtAl_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MomentumIterativeMethod > adv_ensemble_acc_MomentumIterativeMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=BasicIterativeMethod > adv_ensemble_acc_BasicIterativeMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MadryEtAl > adv_ensemble_acc_MadryEtAl_model3_lamda1p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MomentumIterativeMethod > adv_ensemble_acc_MomentumIterativeMethod_model3_lamda1p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u cifar10_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=BasicIterativeMethod > adv_ensemble_acc_BasicIterativeMethod_model3_lamda1p0_loglamda0p5.out 2>&1 &


#Adv transfer acc
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP-adv_transfer.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MadryEtAl --eps=0.04 > adv_transfer_acc_MadryEtAl_models3_eps0p04_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP-adv_transfer.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MomentumIterativeMethod --eps=0.04 > adv_transfer_acc_MomentumIterativeMethod_models3_eps0p04_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP-adv_transfer.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MadryEtAl --eps=0.04 > adv_transfer_acc_MadryEtAl_models3_eps0p04_lamda1p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar10_resnet_EE_DPP-adv_transfer.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MomentumIterativeMethod --eps=0.04 > adv_transfer_acc_MomentumIterativeMethod_models3_eps0p04_lamda1p0_loglamda0p5.out 2>&1 &


#Adv transfer acc (target)
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar10_resnet_EE_DPP-adv_transfer_target.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MadryEtAl --eps=0.04 > adv_transfer_acc_MadryEtAl_models3_eps0p04_lamda2p0_loglamda0p5_target.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP-adv_transfer_target.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=159 --baseline_epoch=139 --attack_method=MomentumIterativeMethod --eps=0.04 > adv_transfer_acc_MomentumIterativeMethod_models3_eps0p04_lamda2p0_loglamda0p5_target.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar10_resnet_EE_DPP-adv_transfer_target.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MadryEtAl --eps=0.04 > adv_transfer_acc_MadryEtAl_models3_eps0p04_lamda1p0_loglamda0p5_target.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar10_resnet_EE_DPP-adv_transfer_target.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch=162 --baseline_epoch=139 --attack_method=MomentumIterativeMethod --eps=0.04 > adv_transfer_acc_MomentumIterativeMethod_models3_eps0p04_lamda1p0_loglamda0p5_target.out 2>&1 &










#Adv Eval
CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=0.0 --log_det_lamda=0.0 --num_models=2 --augmentation=True --epoch=153 --attack_method=MadryEtAl --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=2 --augmentation=True --epoch=153 --attack_method=MadryEtAl --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch=180 --attack_method=MadryEtAl --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=4 --augmentation=True --epoch=170 --attack_method=MadryEtAl --eps=0.05



CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=0.0 --log_det_lamda=0.0 --num_models=2 --augmentation=True --epoch=153 --attack_method=MomentumIterativeMethod --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=2 --augmentation=True --epoch=153 --attack_method=MomentumIterativeMethod --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch=180 --attack_method=MomentumIterativeMethod --eps=0.05

CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_EE_DPP-adveval.py --lamda=2.0 --log_det_lamda=0.3 --num_models=4 --augmentation=True --epoch=170 --attack_method=MomentumIterativeMethod --eps=0.05
