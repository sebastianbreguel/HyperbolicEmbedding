import tensorflow as tf
from tensorflow import keras

from .util import activation_result, poincare_linear


class LinearHyperbolic(keras.layers.Layer):
    """
    Implementation of a hyperbolic linear layer for a neural network, that inherits from the keras Layer class

    Gane et al(2018 version) https://arxiv.org/pdf/1805.09112.pdf
    """

    def __init__(
        self,
        units,
        manifold,
        c,
        activation=None,
        use_bias=True,
        input_hyp=False,
        output_hyp=False,
    ):
        super().__init__()
        self.units = units
        self.c = tf.Variable([c], dtype="float32")
        self.manifold = manifold
        self.activation = keras.activations.get(activation)
        self.has_act = activation is not None
        self.use_bias = use_bias
        self.input_hyp = input_hyp
        self.output_hyp = output_hyp

    def build(self, batch_input_shape):
        #  add seed to random initialization
        w_init = tf.random_normal_initializer(seed=21)
        self.kernel = tf.Variable(
            initial_value=w_init(
                shape=(batch_input_shape[-1], self.units), dtype="float32"
            ),
            dtype="float32",
            trainable=True,
            # seed
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units),
                initializer="zeros",
                dtype=tf.float32,
                trainable=True,
            )

        super().build(batch_input_shape)  # must be at the end

    def call(self, inputs):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        inputs = tf.cast(inputs, tf.float32)

        # apply dropout to kernel

        # if input is not in hyperbolic space, project it to the hyperbolic space
        if self.input_hyp == False:
            inputs = self.manifold.proj(self.manifold.expmap0(inputs, self.c), self.c)

        mv = self.manifold.mobius_matvec(self.kernel, inputs, self.c)
        res = self.manifold.proj(mv, c=self.c)

        if self.use_bias:
            # project the bias to hyperbolic space, then add and finaly project the result to the hyperbolic space

            hyp_bias = self.manifold.expmap0(self.bias, c=self.c)
            hyp_bias = self.manifold.proj(hyp_bias, c=self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, c=self.c)

        if self.has_act:
            res = activation_result(self.manifold, self.activation, res, self.c)

        if self.output_hyp == False:
            res = self.manifold.logmap0(res, c=self.c)

        return res

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "manifold": self.manifold,
            "curvature": self.c,
        }


class LinearHyperbolicPlusPlus(keras.layers.Layer):
    """
    Implementation of a hyperbolic linear layer for a neural network, that inherits from the keras Layer class

    Hyperbolic Neuronal Networks++ version https://arxiv.org/pdf/2006.08195.pdf
    """

    def __init__(
        self,
        units,
        manifold,
        c,
        activation=None,
        use_bias=True,
        input_hyp=False,
        output_hyp=False,
        gain=1.0,
    ):
        super().__init__()
        self.units = units
        self.c = tf.Variable([c], dtype="float32")
        self.manifold = manifold
        self.activation = keras.activations.get(activation)
        self.has_act = activation is not None
        self.use_bias = use_bias
        self.input_hyp = input_hyp
        self.output_hyp = output_hyp
        self.gain = gain

    def build(self, batch_input_shape):
        weight = tf.Variable(
            tf.random.normal(
                shape=(batch_input_shape[-1], self.units),
                mean=0,
                stddev=(2 * batch_input_shape[-1] * self.units / 1) ** -0.5 * self.gain,
                dtype="float32",
                seed=21,
            ),
            trainable=True,
        )

        self.weigh_v = tf.Variable(
            initial_value=weight,
            dtype="float32",
            trainable=True,
        )

        self.weight_g = tf.Variable(
            # norm of weight
            initial_value=tf.norm(weight, axis=0, keepdims=True),
            dtype="float32",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units),
                initializer="zeros",
                dtype=tf.float32,
                trainable=True,
            )

        super().build(batch_input_shape)  # must be at the end

    def call(self, inputs):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """

        inputs = tf.cast(inputs, tf.float32)

        if self.input_hyp == False:
            inputs = self.manifold.proj(self.manifold.expmap0(inputs, self.c), self.c)

        v_norm = tf.clip_by_value(
            tf.norm(self.weigh_v, axis=0, keepdims=True),
            clip_value_min=1e-6,
            clip_value_max=1e6,
        )
        weigh_v_norm = self.weigh_v / v_norm
        x = poincare_linear(
            inputs, self.manifold, self.weight_g, weigh_v_norm, self.bias, self.c
        )

        if self.has_act:
            x = activation_result(self.manifold, self.activation, x, self.c)

        if self.output_hyp == False:
            x = self.manifold.logmap0(x, c=self.c)

        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "manifold": self.manifold,
            "curvature": self.c,
        }
