package com.dark.sky.nn

import breeze.linalg.{DenseMatrix, rand}
import breeze.numerics.sqrt

class LayerChain(inLayerSize: Int, outLayerSize: Int)(implicit activation: ActivationFunction) {
  private val epsilon = LayerChain.SQRT6 / sqrt(inLayerSize + outLayerSize)

  lazy val weights: DenseMatrix[Double] = LayerChain.initWeights(inLayerSize, outLayerSize, epsilon)

  def apply(weight: DenseMatrix[Double]): Activation = new Activation {
    private val weightT: DenseMatrix[Double] = weight.t

    override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] =
      activation(x * weightT)
  }
}

private object LayerChain {
  private val SQRT6: Double = sqrt(6)

  private def initWeights(inSize: Int, outSize: Int, epsilon: Double): DenseMatrix[Double] = {
    DenseMatrix.fill(outSize, inSize + 1)(rand() * 2 * epsilon - epsilon)
  }
}
