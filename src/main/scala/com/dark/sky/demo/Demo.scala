package com.dark.sky.demo

import java.util.function.{Function, Supplier}

import breeze.linalg.{*, DenseMatrix, randomInt}
import breeze.numerics.sqrt
import com.dark.sky.demo.draw.DemoDrawer
import com.dark.sky.nn.NeuralNetwork

object Demo extends App {
  val (t, cv, test) = TrainUtil.readTrainingAndCrossValidationAndTestData()
  val nn = NeuralNetwork(400)(25)(10).train(t._1, t._2)
  val nnAccuracy = checkAccuracy(nn, cv)
  println(s"Training accuracy without regularization: ${checkAccuracy(nn, t)}")
  println(s"Cross Validation accuracy without regularization: $nnAccuracy")

  val (mostAccurate, lambda, accuracy) = lambdas.foldLeft (nn, 0.0, nnAccuracy) ((current, lambda) => {
    val newNN = current._1.train(t._1, t._2, lambda)
    val accuracy = checkAccuracy(newNN, cv)
    println(s"Training accuracy with lambda $lambda: ${checkAccuracy(newNN, t)}")
    println(s"Cross Validation accuracy with lambda $lambda: $accuracy")
    if (accuracy < current._3) current else (newNN, lambda, accuracy)
  })
  println(s"Cross Validation accuracy of most accurate (lambda is $lambda): $accuracy")
  println(s"Test accuracy of most accurate: ${checkAccuracy(nn, test)}")

  private def lambdas: List[Double] =
    Stream.iterate(0.01, 7)(_ * sqrt(10)).toList

  private def checkAccuracy(nn: NeuralNetwork, data: (DenseMatrix[Double], DenseMatrix[Int])): Double = {
    var i = 0
    var r = 0
    data._1(*, ::) foreach (row => {
      if (nn.predict(row.toDenseMatrix) == data._2.valueAt(i)) r += 1
      i += 1
    })
    r.toDouble / i
  }

  new DemoDrawer(new Function[java.util.List[java.lang.Double], java.lang.Integer] {
    override def apply(x: java.util.List[java.lang.Double]): java.lang.Integer = {
      val iterator = x.iterator()
      mostAccurate.predict(DenseMatrix.fill[Double](1, x.size())(iterator.next()))
    }
  }, new Supplier[java.util.List[java.lang.Double]] {
    override def get(): java.util.List[java.lang.Double] = {
      val l = new java.util.ArrayList[java.lang.Double](400)
      test._1(randomInt(1, (0, 999)).apply(0), ::).inner foreach (l.add(_))
      l
    }
  })
}
