"""
Perceptron package.

This package provides a simple Perceptron implementation,
adapted from Raschka et al.'s *Machine Learning with PyTorch and Scikit-Learn*,
with an additional early stopping criterion for convergence.

Modules
-------
perceptron : Contains the Perceptron class.
"""

__author__ = "Lucas Campagnaro"

from .perceptron import Perceptron

__all__ = ["Perceptron"]