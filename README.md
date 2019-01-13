# Improving Adversarial Robustness via Promoting Ensemble Diversity
## Anonymous Authors

Code for reproducing most of the results of our ICML submission with paper ID:().  
Improving Adversarial Robustness via Promoting Ensemble Diversity


## Envoronment settings and libs we used in our experiments

This project is tested under the following environment setting.
- OS: Ubuntu 16.04.3
- GPU: Geforce 1080 Ti or Titan X (Pascal or Maxwell)
- Cuda: 9.0, Cudnn: v7.03
- Python: 2.7.12
- cleverhans: 2.1.0
- Keras: 2.2.4
- tensorflow-gpu: 1.9.0

Thank the authors of these libs. We also thank the authors of [keras-resnet](https://github.com/raghakot/keras-resnet) for providing their code. Our code is widely adapted from their repositories.

In the following, we first provide the code for training our proposed methods and baselines. After that, the evaluation code, such as attacking, is provided.

## Training codes.

### Training baselines and ADP.

For training on MNIST dataset, 
```python
python -u mnist_resnet_EE_DPP.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True 
```
where the baseline is implemented with alpha_value = beta_value = 0, and the ADP is implemented with the corresponding value in Table 1.

For CIFAR10 and CIFAR100, the command is similar, with following:
```python
# CIFAR10
python -u cifar10_resnet_EE_DPP.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True
# CIFAR100
python -u cifar100_resnet_EE_DPP.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True
```

Using the aboved command, the model used in Tab.1 & 2 & 3 can be reproduced.

### Adversarial Training w/o ADP
For adversarial training, we use FGSM and PGD methods to construct adversarial examples.
The model can be trained using the following commands:

```python
# PGD + ADP
python -u cifar10_resnet_EE_DPP_adv.py --attack_method=MadryEtAl --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True 
# PGD without ADP
python -u cifar10_resnet_EE_DPP_adv.py --attack_method=MadryEtAl --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True  

# FGSM + ADP
python -u cifar10_resnet_EE_DPP_adv.py --attack_method=FastGradientMethod --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True 
# FGSM without ADP
python -u cifar10_resnet_EE_DPP_adv.py --attack_method=FastGradientMethod --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True 
```

The model used in Tab. 4 can be reproduced.


## Evaluation codes.

### Tab.1 
```bash
python -u [dataset]_resnet_EE_DPP-eval_nor_acc.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch]
```
By substituting the corresponding parameters in the aboved command line, the accuracy can be reproduced in Tab 1.
The ```checkpoint_epoch``` indicates the corresponding checkpoint file which needs to be tested.
```dataset``` can be ```mnist, cifar10, cifar100```.

### Tab. 2 & Tab. 3.
We test our model using both non-iterative attacking methods, including DeepFool, CarliniWagnerL2, ElasticNetMethod, SPSA, LBFGS, JSMA. TODO: most of them are iterative attack methods?

Our model can be tested using the following command:
```python
python -u [dataset]_resnet_EE_DPP-adv_ensemble_eval_noniterative.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --attack_method=[attack_method]
```

Note that for JSMA, the attack algorithm provided by cleverhans is not useable. We implement it ourself, which can be used in the following command.
```python
python -u [dataset]_resnet_EE_DPP-adv_ensemble_eval_jsma.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --attack_method=[attack_method]
```

For iterative attack methods, our model can be tested using:
```python
python -u [dataset]_resnet_EE_DPP-adv_ensemble_eval.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[baseline_checkpoint_epoch] --attack_method=[attack_method]
```
In this part, ADP and baseline methods are tested together.

### Fig 3.

```python
# Untarget method:
python -u cifar10_resnet_EE_DPP-adv_transfer.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[baseline_checkpoint_epoch] --attack_method=[attack_method]

python -u cifar10_resnet_EE_DPP-adv_transfer_target.py -lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[baseline_checkpoint_epoch] --attack_method=[attack_method]
```
For some attack methods, ```--epsilon``` is required to specify the scale for adversarial examples.

### Detect metrics.
```python 
python cifar10_resnet_EE_DPP-adveval.py -lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[baseline_checkpoint_epoch] --attack_method=[attack_method]
```