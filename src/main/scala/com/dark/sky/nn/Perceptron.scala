package com.dark.sky.nn

import breeze.linalg.{DenseMatrix, rand}
import breeze.numerics.sqrt

class Perceptron(inSize: Int, outSize: Int)(implicit activation: ActivationFunction) {
  private val epsilon = Perceptron.SQRT6 / sqrt(inSize + outSize)

  lazy val weight: DenseMatrix[Double] = Perceptron.initWeight(inSize, outSize, epsilon)

  def apply(weight: DenseMatrix[Double]): Activation = new Activation {
    private val weightT: DenseMatrix[Double] = weight.t

    override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] =
      activation(x * weightT)
  }
}

private object Perceptron {
  private val SQRT6: Double = sqrt(6)

  private def initWeight(inSize: Int, outSize: Int, epsilon: Double): DenseMatrix[Double] = {
    DenseMatrix.fill(outSize, inSize + 1)(rand() * 2 * epsilon - epsilon)
  }
}
