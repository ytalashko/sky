package com.dark.sky.nn

import breeze.linalg.DenseMatrix

trait Activation {
  def apply(x: DenseMatrix[Double]): DenseMatrix[Double]
}
