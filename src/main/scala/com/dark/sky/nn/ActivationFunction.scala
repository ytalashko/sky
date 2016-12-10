package com.dark.sky.nn

import breeze.linalg.DenseMatrix
import breeze.numerics.{exp, pow}

trait ActivationFunction extends Activation {
  def gradient(x: DenseMatrix[Double]): DenseMatrix[Double]
}

object Sigmoid extends ActivationFunction {
  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    pow(exp(-x) + 1.0, -1)
  }
  override def gradient(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sigmoid = Sigmoid(x)
    sigmoid :* (-sigmoid + 1.0)
  }
}
