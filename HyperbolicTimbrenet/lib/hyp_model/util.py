
import tensorflow as tf
from  .manifold import  cosh ,  sinh ,  asinh

def poincare_linear(x, manifold, weight_g, weight_v, bias, c):
    rc = tf.sqrt(c)
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    x = sinh(rc * x) / rc
    
    pow_sum = tf.math.reduce_sum(tf.math.pow(x, 2), axis=-1, keepdims=True)
    return manifold.proj(x / (1 +  tf.math.sqrt(1 + c * pow_sum)), c)

def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    rc = tf.math.sqrt(c)
    drcr = 2.0 * r * rc

    rcx = rc * x
    cx2 = tf.math.reduce_sum(tf.math.pow(rcx, 2), axis=-1, keepdims=True)

    first_part = 2 * z_norm / rc
    second = 2 * (rcx @ z_unit) * cosh(drcr)
    third = (1 + cx2) * sinh(drcr)
    forth = tf.clip_by_value(
        1 - cx2, clip_value_min=1e-6, clip_value_max=1e6
    )
    five = second + third
    six = tf.math.divide(five, forth)
    seven = first_part * asinh(six)
    return seven




def activation_result(manifold, activation, input, c=1.0):
    result = manifold.logmap0(input, c=c)
    result = activation(result)
    result = manifold.expmap0(result, c=c)
    result = manifold.proj(result, c=c)
    return result
