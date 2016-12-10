package com.dark.sky.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import breeze.optimize.{DiffFunction, LBFGS}

trait Trainable {
  def train(x: DenseMatrix[Double], y: DenseMatrix[Int], lambda: Double = 0): NeuralNetwork
}

trait NeuralNetwork extends Trainable {
  def predict(x: DenseMatrix[Double]): Int
}

private class Skeleton(layers: List[Int])(implicit activation: ActivationFunction) extends Trainable {
  private val perceptrons: List[Perceptron] =
    layers zip layers.tail map (chain => new Perceptron(chain._1, chain._2))

  override def train(x: DenseMatrix[Double], y: DenseMatrix[Int], lambda: Double): NeuralNetwork = {
    val diff: DenseVector[Double] => (Double, DenseVector[Double]) = differentiable(x, y, lambda)
    val diffFunction: DiffFunction[DenseVector[Double]] = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = diff(x)
    }
//    TODO: add a way to configure minimization options.
    val minimizer = new LBFGS[DenseVector[Double]](maxIter = 500, m = 7, tolerance = 1E-12)
    val optimumWeights = minimizer.minimize(diffFunction,
      perceptrons map (_.weight) map (_.toDenseVector) reduce (DenseVector.vertcat(_, _)))
    new Predictor(this, activations(optimumWeights))
  }

  private def differentiable(x: DenseMatrix[Double], y: DenseMatrix[Int], lambda: Double)
                            (weightsVec: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val weights = reshape(weightsVec)

    val zaPairs = weights.foldLeft(List(x -> Skeleton.withBias(x))) ((zaPairs, weight) => {
      val z = zaPairs.head._2 * weight.t
      val a = activation(z)
      z -> (if (zaPairs.size == weights.size) a else Skeleton.withBias(a)) :: zaPairs
    })
    val aVec = zaPairs.head._2.toDenseVector

    val yMatrix = DenseMatrix.tabulate[Int](y.rows, layers.last) ((r, c) => if (y.valueAt(r, 0) == c) 1 else 0)
    val yVec = (yMatrix.toDenseVector map (_.toDouble)).t

    val m = y.rows
    val regWeights = weights map Skeleton.recallBias
    val regWeightsVec = weights map (_(::, 1 to -1)) map (_.toDenseVector) reduce (DenseVector.vertcat(_, _))
    val j = -1.0 / m * (yVec * log(aVec) + ((-yVec + 1.0) * log(-aVec + 1.0))) + lambda / (2 * m) * (regWeightsVec.t * regWeightsVec)

    val reversedWeights = weights.reverse
    val gradients = zaPairs.slice(1, zaPairs.size - 1).foldLeft(List(zaPairs.head._2 - (yMatrix map (_.toDouble)))) ((deltas, za) => {
      val i = deltas.size
      val delta = deltas.head * reversedWeights(i - 1)(::, 1 to -1) :* activation.gradient(za._1)
      delta :: deltas
    }).zipWithIndex zip regWeights map (deltaWithIndex => {
      (deltaWithIndex._1._1.t * zaPairs(zaPairs.size - 1 - deltaWithIndex._1._2)._2 + (deltaWithIndex._2 :* lambda)) :/ m.toDouble
    })

    (j, gradients map (_.toDenseVector) reduce (DenseVector.vertcat(_, _)))
  }

  private def activations(weights: DenseVector[Double]): List[Activation] = {
    perceptrons zip reshape(weights) map (pair => pair._1(pair._2))
  }

  private def reshape(weights: DenseVector[Double]): List[DenseMatrix[Double]] = {
    (layers map (_ + 1) zip layers.tail).foldLeft(0 -> List[DenseMatrix[Double]]()) ((acc, e) => {
      val count: Int = acc._1 + e._1 * e._2
      count -> (weights.slice(acc._1, count).toDenseMatrix.reshape(e._2, e._1) :: acc._2)
    })._2.reverse
  }
}

private class Predictor(skeleton: Skeleton, activations: List[Activation]) extends NeuralNetwork {
  override def predict(x: DenseMatrix[Double]): Int =
    activations.foldLeft(x) ((a, activation) => activation(Skeleton.withBias(a))).toArray.zipWithIndex.maxBy(_._1)._2

  override def train(x: DenseMatrix[Double], y: DenseMatrix[Int], lambda: Double): NeuralNetwork =
    skeleton.train(x, y, lambda)
}

private object Skeleton {
  def withBias(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val r = DenseMatrix.ones[Double](x.rows, x.cols + 1)
    r(::, 1 to -1) := x
    r
  }

  private def recallBias(weight: DenseMatrix[Double]): DenseMatrix[Double] = {
    val r = DenseMatrix.zeros[Double](weight.rows, weight.cols)
    r(::, 1 to -1) := weight(::, 1 to -1)
    r
  }
}

object NeuralNetwork {
  implicit val DEFAULT_ACTIVATION: ActivationFunction = Sigmoid

//  TODO: add a way to provide custom Activation Functions
  def apply(inSize: Int)(sizes: Int*)(outSize: Int): Trainable =
    new Skeleton((inSize :: sizes.toList) :+ outSize)
}
