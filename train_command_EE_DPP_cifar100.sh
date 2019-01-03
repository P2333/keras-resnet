# jungpu8
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p3_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.1 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p1_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.2 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p2_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda2p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP.py --lamda=1.5 --log_det_lamda=0.5 --num_models=3 --augmentation=True > ../cifar100_resnet_EE_DPP_models3_lamda1p5_logdetlamda0p5_True.out 2>&1 &




CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.5 --num_models=6 --augmentation=True > ../cifar100_resnet_EE_DPP_models6_lamda2p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=6 --augmentation=True > ../cifar100_resnet_EE_DPP_models6_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP.py --lamda=2.0 --log_det_lamda=0.0 --num_models=6 --augmentation=True > ../cifar100_resnet_EE_DPP_models6_lamda2p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda3p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP.py --lamda=3.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda3p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=0.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP.py --lamda=4.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda4p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP.py --lamda=5.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda5p0_logdetlamda0p5_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda3p0_logdetlamda1p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=4.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda4p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP.py --lamda=5.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda5p0_logdetlamda0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP.py --lamda=3.0 --log_det_lamda=2.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda3p0_logdetlamda2p0_True.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP.py --lamda=3.0 --log_det_lamda=3.0 --num_models=9 --augmentation=True > ../cifar100_resnet_EE_DPP_models9_lamda3p0_logdetlamda3p0_True.out 2>&1 &

# Nor acc eval
CUDA_VISIBLE_DEVICES=1 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095'

CUDA_VISIBLE_DEVICES=4 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='082'

CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.2 --num_models=3 --augmentation=True --epoch='088'

CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.1 --num_models=3 --augmentation=True --epoch='084'

CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='085'

CUDA_VISIBLE_DEVICES=4 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135'

CUDA_VISIBLE_DEVICES=3 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='131'



CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=2.0 --log_det_lamda=0.5 --num_models=6 --augmentation=True --epoch='165'

CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=0.0 --log_det_lamda=0.0 --num_models=6 --augmentation=True --epoch='148'



CUDA_VISIBLE_DEVICES=0 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172'

CUDA_VISIBLE_DEVICES=2 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=0.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='094'

CUDA_VISIBLE_DEVICES=0 python cifar100_resnet_EE_DPP-eval_nor_acc.py --lamda=3.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='083'


#Adv ensemble acc (noniterative)
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model3_lamda1p0_loglamda0p3_CWconstant0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='082' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model3_lamda0_loglamda0_CWconstant0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='085' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model3_lamda1p0_loglamda0_CWconstant0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model3_lamda2p0_loglamda0p5_CWconstant0p01.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=1.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='131' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model3_lamda1p0_loglamda0p5_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=2.0 --log_det_lamda=0.5 --num_models=6 --augmentation=True --epoch='165' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model6_lamda2p0_loglamda0p5_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=0.0 --log_det_lamda=0.0 --num_models=6 --augmentation=True --epoch='148' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model6_lamda0_loglamda0_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda3p0_loglamda0p5_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=0.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='094' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda0_loglamda0_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=5.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='143' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda5p0_loglamda0p5_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=4.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='139' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda4p0_loglamda0p5_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda3p0_loglamda1p0_CWconstant0p001.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=3.0 --log_det_lamda=3.0 --num_models=9 --augmentation=True --epoch='128' --attack_method=CarliniWagnerL2 > cifar100_adv_ensemble_acc_CarliniWagnerL2_model9_lamda3p0_loglamda3p0_CWconstant0p001.out 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095' --attack_method=ElasticNetMethod > cifar100_adv_ensemble_acc_ElasticNetMethod_model3_lamda1p0_loglamda0p3_constant5p0_beta0p01.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='082' --attack_method=ElasticNetMethod > cifar100_adv_ensemble_acc_ElasticNetMethod_model3_lamda0_loglamda0_constant5p0_beta0p01.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=1.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='085' --attack_method=ElasticNetMethod > cifar100_adv_ensemble_acc_ElasticNetMethod_model3_lamda1p0_loglamda0_constant5p0_beta0p01.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --attack_method=ElasticNetMethod > cifar100_adv_ensemble_acc_ElasticNetMethod_model9_lamda3p0_loglamda0p5_constant1p0_beta0p01.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=0.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='094' --attack_method=ElasticNetMethod > cifar100_adv_ensemble_acc_ElasticNetMethod_model9_lamda0_loglamda0_constant1p0_beta0p01.out 2>&1 &




CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' > cifar100_adv_ensemble_acc_JSMA_model9_lamda3p0_loglamda1p0_gamma0p05_eps0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=3.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='083' > cifar100_adv_ensemble_acc_JSMA_model9_lamda3p0_loglamda0_gamma0p05_eps0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=0.0 --log_det_lamda=0.0 --num_models=9 --augmentation=True --epoch='094' > cifar100_adv_ensemble_acc_JSMA_model9_lamda0_loglamda0_gamma0p05_eps0p1.out 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' > cifar100_adv_ensemble_acc_JSMA_model3_lamda2p0_loglamda0p5_gamma0p05_eps0p1.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --epoch='082' > cifar100_adv_ensemble_acc_JSMA_model3_lamda0_loglamda0_gamma0p05_eps0p1.out 2>&1 &


#Adv ensemble acc
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095' --baseline_epoch='082' --attack_method=MadryEtAl > cifar100_adv_ensemble_acc_MadryEtAl_model3_lamda1p0_loglamda0p3.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095' --baseline_epoch='082' --attack_method=MomentumIterativeMethod > cifar100_adv_ensemble_acc_MomentumIterativeMethod_model3_lamda1p0_loglamda0p3.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=1.0 --log_det_lamda=0.3 --num_models=3 --augmentation=True --epoch='095' --baseline_epoch='082' --attack_method=BasicIterativeMethod > cifar100_adv_ensemble_acc_BasicIterativeMethod_model3_lamda1p0_loglamda0p3.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=MadryEtAl > AvergeStep_cifar100_adv_ensemble_acc_MadryEtAl_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=MomentumIterativeMethod > AvergeStep_cifar100_adv_ensemble_acc_MomentumIterativeMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=BasicIterativeMethod > AvergeStep_cifar100_adv_ensemble_acc_BasicIterativeMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=FastGradientMethod > cifar100_adv_ensemble_acc_FastGradientMethod_model3_lamda2p0_loglamda0p5.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --baseline_epoch='094' --attack_method=MomentumIterativeMethod > cifar100_adv_ensemble_acc_MomentumIterativeMethod_model9_lamda3p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --baseline_epoch='094' --attack_method=BasicIterativeMethod > cifar100_adv_ensemble_acc_BasicIterativeMethod_model9_lamda3p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --baseline_epoch='094' --attack_method=MadryEtAl > cifar100_adv_ensemble_acc_MadryEtAl_model9_lamda3p0_loglamda0p5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='172' --baseline_epoch='094' --attack_method=FastGradientMethod > cifar100_adv_ensemble_acc_FastGradientMethod_model9_lamda3p0_loglamda0p5.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' --baseline_epoch='094' --attack_method=MomentumIterativeMethod > cifar100_adv_ensemble_acc_MomentumIterativeMethod_model9_lamda3p0_loglamda1p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' --baseline_epoch='094' --attack_method=BasicIterativeMethod > cifar100_adv_ensemble_acc_BasicIterativeMethod_model9_lamda3p0_loglamda1p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' --baseline_epoch='094' --attack_method=MadryEtAl > cifar100_adv_ensemble_acc_MadryEtAl_model9_lamda3p0_loglamda1p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=3.0 --log_det_lamda=1.0 --num_models=9 --augmentation=True --epoch='166' --baseline_epoch='094' --attack_method=FastGradientMethod > cifar100_adv_ensemble_acc_FastGradientMethod_model9_lamda3p0_loglamda1p0.out 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval.py --lamda=5.0 --log_det_lamda=0.5 --num_models=9 --augmentation=True --epoch='143' --baseline_epoch='094' --attack_method=MomentumIterativeMethod > cifar100_adv_ensemble_acc_MomentumIterativeMethod_model9_lamda5p0_loglamda0p5.out 2>&1 &



#Adv ensemble acc(target)
CUDA_VISIBLE_DEVICES=0 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_target.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=MadryEtAl > cifar100_adv_ensemble_acc_MadryEtAl_model3_lamda2p0_loglamda0p5_target.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_target.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=MomentumIterativeMethod > cifar100_adv_ensemble_acc_MomentumIterativeMethod_model3_lamda2p0_loglamda0p5_target.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u cifar100_resnet_EE_DPP-adv_ensemble_eval_target.py --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --epoch='135' --baseline_epoch='082' --attack_method=BasicIterativeMethod > cifar100_adv_ensemble_acc_BasicIterativeMethod_model3_lamda2p0_loglamda0p5_target.out 2>&1 &

