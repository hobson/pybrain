__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid, steep_sigmoid


class SigmoidLayer(NeuronLayer):
    """Layer implementing the sigmoid squashing function."""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outbuf * (1 - outbuf) * outerr


class SteepSigmoidLayer(NeuronLayer):
    """Layer implementing a steeper logistic function, a sigmoidal squashing function.

    Useful for bayesean classifiers because is approaches 1 and 0 more rapidly.

    ((1. + exp(-x))^-1)^s

    Where `s` is a exponent that increases the "steepness" when `s` > 1
    """
    exponent = 2

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = steep_sigmoid(inbuf, exponent=self.exponent)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """The chain rule means we need to multiply by another steep_sigmoid evaluation (outbuf)

        Chain rule for derivative of outer exponent gives:

        self.exponent * (outbuf ** (self.exponent - 1)) * outbuf * (1 - outbuf) * outerr

        Which simplifies to:

        self.exponent * (outbuf ** self.exponent) * (1 - outbuf) * outerr
        """
        inerr[:] = self.exponent * (outbuf ** self.exponent) * (1 - outbuf) * outerr
