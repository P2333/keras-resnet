from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from cleverhans.attacks_tf import _project_perturbation, UnrolledAdam
from cleverhans.attacks import Attack

# Training parameters
num_classes = 10
log_offset = 1e-20
det_offset = 1e-6


np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lamda', 1.0, "lamda for Ensemble Entropy(EE)")
tf.app.flags.DEFINE_float('nonME_lamda', 0.5, "lamda for non-ME")
tf.app.flags.DEFINE_float('log_det_lamda', 0.5, "lamda for non-ME")

tf.app.flags.DEFINE_bool('augmentation', False, "whether use data augmentation")
tf.app.flags.DEFINE_integer('num_models', 2, "The num of models in the ensemble")
tf.app.flags.DEFINE_integer('epoch', 1, "epoch of the checkpoint to load")

tf.app.flags.DEFINE_string('attack_method', 'MadryEtAl', "FastGradientMethod, MadryEtAl")
tf.app.flags.DEFINE_float('eps', 0.05, "maximal eps for attacks")
tf.app.flags.DEFINE_integer('baseline_epoch', 1, "epoch of the checkpoint to load")
tf.app.flags.DEFINE_integer('batch_size', 64, "")
tf.app.flags.DEFINE_float('label_smooth', 1.0, "")
tf.app.flags.DEFINE_float('eps_', 0.01, "eps for iterative attacks")
tf.app.flags.DEFINE_float('param', 0.01, "params for non-iterative attacks")


zero = tf.constant(0, dtype=tf.float32)
## Functions ##
def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)

def Ensemble_Entropy(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble

def non_ME(y_true, y_pred, num_model=FLAGS.num_models):
    ONE = tf.ones_like(y_true) # batch_size X (num_class X num_models)
    R_y_true = ONE - y_true
    non_y_pred = tf.multiply(R_y_true, y_pred)
    y_p = tf.split(non_y_pred, num_model, axis=-1)
    sum_nonme = 0
    for i in range(num_model):
        sum_nonme += Entropy(y_p[i] / tf.reduce_sum(y_p[i], axis=-1, keepdims=True))
    return sum_nonme

def log_det(y_true, y_pred, num_model=FLAGS.num_models):
    bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det


## Metrics ##
def Ensemble_Entropy_metric(y_true, y_pred, num_model=FLAGS.num_models):
    EE = Ensemble_Entropy(y_true, y_pred, num_model=num_model)
    return K.mean(EE)

def non_ME_metric (y_true, y_pred, num_model=FLAGS.num_models):
    sum_nonme = non_ME(y_true, y_pred, num_model)
    return sum_nonme / num_model

def log_det_metric(y_true, y_pred, num_model=FLAGS.num_models):
    log_dets = log_det(y_true, y_pred, num_model=num_model)
    return K.mean(log_dets)

def acc_metric(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    acc = 0
    for i in range(num_model):
        acc += keras.metrics.categorical_accuracy(y_t[i], y_p[i])
    return acc / num_model

def Loss_withEE_plus(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    sum_nonme = non_ME(y_true, y_pred, num_model)
    return CE_all - FLAGS.lamda * EE + FLAGS.nonME_lamda * sum_nonme

def Loss_withEE_plus_leave_one(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    sum_nonme = non_ME(y_true, y_pred, num_model - 1)###leave one model
    return CE_all - FLAGS.lamda * EE + FLAGS.nonME_lamda * sum_nonme

def Loss_withEE_DPP(y_true, y_pred, num_model=FLAGS.num_models, label_smooth=FLAGS.label_smooth):
    scale = (1 - label_smooth) / (num_classes * label_smooth - 1)
    y_t_ls = scale * tf.ones_like(y_true) + y_true
    y_t_ls = (num_model * y_t_ls) / tf.reduce_sum(y_t_ls, axis=1, keepdims=True) 
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_t_ls, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    EE = Ensemble_Entropy(y_true, y_pred, num_model)
    log_dets = log_det(y_true, y_pred, num_model)
    return CE_all - FLAGS.lamda * EE - FLAGS.log_det_lamda * log_dets


## SPSA attack
def pgd_attack(loss_fn, input_image, label, epsilon, num_steps,
               optimizer=UnrolledAdam(),
               project_perturbation=_project_perturbation,
               early_stop_loss_threshold=None,
               is_debug=False):
    """Projected gradient descent for generating adversarial images.

    Args:
        :param loss_fn: A callable which takes `input_image` and `label` as
                        arguments, and returns a batch of loss values. Same
                        interface as UnrolledOptimizer.
        :param input_image: Tensor, a batch of images
        :param label: Tensor, a batch of labels
        :param epsilon: float, the L-infinity norm of the maximum allowable
                                        perturbation
        :param num_steps: int, the number of steps of gradient descent
        :param optimizer: An `UnrolledOptimizer` object
        :param project_perturbation: A function, which will be used to enforce
                                     some constraint. It should have the same
                                     signature as `_project_perturbation`.
        :param early_stop_loss_threshold: A float or None. If specified, the
                                          attack will end if the loss is below
                                          `early_stop_loss_threshold`.
        :param is_debug: A bool. If True, print debug info for attack progress.

    Returns:
        adversarial version of `input_image`, with L-infinity difference less
            than epsilon, which tries to minimize loss_fn.

    Note that this function is not intended as an Attack by itself. Rather, it
    is designed as a helper function which you can use to write your own attack
    methods. The method uses a tf.while_loop to optimize a loss function in
    a single sess.run() call.
    """

    init_perturbation = tf.random_uniform(tf.shape(input_image),
                                          minval=-epsilon, maxval=epsilon,
                                          dtype=tf_dtype)
    init_perturbation = project_perturbation(init_perturbation,
                                             epsilon, input_image)
    init_optim_state = optimizer.init_state([init_perturbation])
    nest = tf.contrib.framework.nest

    def loop_body(i, perturbation, flat_optim_state):
        """Update perturbation to input image."""
        optim_state = nest.pack_sequence_as(structure=init_optim_state,
                                            flat_sequence=flat_optim_state)

        def wrapped_loss_fn(x):
            return loss_fn(input_image + x, label)
        new_perturbation_list, new_optim_state = optimizer.minimize(
                wrapped_loss_fn, [perturbation], optim_state)
        loss = tf.reduce_mean(wrapped_loss_fn(perturbation), axis=0)
        if is_debug:
            with tf.device("/cpu:0"):
                loss = tf.Print(loss, [loss], "Total batch loss")
        projected_perturbation = project_perturbation(
                new_perturbation_list[0], epsilon, input_image)
        with tf.control_dependencies([loss]):
            i = tf.identity(i)
            if early_stop_loss_threshold:
                i = tf.cond(tf.less(loss, early_stop_loss_threshold),
                            lambda: float(num_steps), lambda: i)
        return i + 1, projected_perturbation, nest.flatten(new_optim_state)

    def cond(i, *_):
        return tf.less(i, num_steps)

    flat_init_optim_state = nest.flatten(init_optim_state)
    _, final_perturbation, _ = tf.while_loop(
        cond,
        loop_body,
        loop_vars=[tf.constant(0.), init_perturbation,
                   flat_init_optim_state],
        parallel_iterations=1,
        back_prop=False)

    adversarial_image = input_image + tf.clip_by_value(final_perturbation,clip_value_min=-epsilon,clip_value_max=epsilon)
    return tf.stop_gradient(adversarial_image)

class SPSA(Attack):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which
    is useful when the model is non-differentiable, or more generally, the
    gradients do not point in useful directions.
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        super(SPSA, self).__init__(model, back, sess, dtypestr)

    def generate(self, x, y=None, y_target=None, epsilon=None, num_steps=None,
                 is_targeted=False, early_stop_loss_threshold=None,
                 learning_rate=0.01, delta=0.01, batch_size=128, spsa_iters=1,
                 is_debug=False):
        """
        Generate symbolic graph for adversarial examples.

        :param x: The model's symbolic inputs. Must be a batch of size 1.
        :param y: A Tensor or None. The index of the correct label.
        :param y_target: A Tensor or None. The index of the target label in a
                         targeted attack.
        :param epsilon: The size of the maximum perturbation, measured in the
                        L-infinity norm.
        :param num_steps: The number of optimization steps.
        :param is_targeted: Whether to use a targeted or untargeted attack.
        :param early_stop_loss_threshold: A float or None. If specified, the
                                          attack will end as soon as the loss
                                          is below `early_stop_loss_threshold`.
        :param learning_rate: Learning rate of ADAM optimizer.
        :param delta: Perturbation size used for SPSA approximation.
        :param batch_size: Number of inputs to evaluate at a single time. Note
                           that the true batch size (the number of evaluated
                           inputs for each update) is `batch_size * spsa_iters`
        :param spsa_iters: Number of model evaluations before performing an
                           update, where each evaluation is on `batch_size`
                           different inputs.
        :param is_debug: If True, print the adversarial loss after each update.
        """
        from cleverhans.attacks_tf import SPSAAdam, margin_logit_loss

        optimizer = SPSAAdam(lr=learning_rate, delta=delta,
                             num_samples=batch_size, num_iters=spsa_iters)

        def loss_fn(x, label):
            logits = self.model.get_logits(x)
            loss_multiplier = 1 if is_targeted else -1
            return loss_multiplier * margin_logit_loss(
                logits, label, num_classes=self.model.num_classes)

        y_attack = y_target if is_targeted else y
        adv_x = pgd_attack(
            loss_fn, x, y_attack, epsilon, num_steps=num_steps,
            optimizer=optimizer,
            early_stop_loss_threshold=early_stop_loss_threshold,
            is_debug=is_debug,
        )
        return adv_x