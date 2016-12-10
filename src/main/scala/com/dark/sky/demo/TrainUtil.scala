package com.dark.sky.demo

import breeze.linalg.{DenseMatrix, randomInt}

import scala.io.Source

object TrainUtil {

//  training/cv/test - 60/20/20 (%)
  def readTrainingAndCrossValidationAndTestData(): ((DenseMatrix[Double], DenseMatrix[Int]),
    (DenseMatrix[Double], DenseMatrix[Int]), (DenseMatrix[Double], DenseMatrix[Int])) = {
    val data = readTrainingData()
    val trainingAndCrossValidationIndexes = 0 until 5000 by 500 flatMap (randomValues(400, _))
    val trainingIndexes = 0 until 4000 by 400 flatMap (floor => floor until floor + 300 map trainingAndCrossValidationIndexes)
    val crossValidationIndexes = trainingAndCrossValidationIndexes filter (!trainingIndexes.contains(_))
    val testIndexes = 0 until 5000 filter (!trainingAndCrossValidationIndexes.contains(_))
    (data._1(trainingIndexes, ::).toDenseMatrix -> data._2(trainingIndexes, ::).toDenseMatrix,
      data._1(crossValidationIndexes, ::).toDenseMatrix -> data._2(crossValidationIndexes, ::).toDenseMatrix,
      data._1(testIndexes, ::).toDenseMatrix -> data._2(testIndexes, ::).toDenseMatrix)
  }

//  training/test - 70/30 (%)
  def readTrainingAndTestData(): ((DenseMatrix[Double], DenseMatrix[Int]),
    (DenseMatrix[Double], DenseMatrix[Int])) = {
    val data = readTrainingData()
    val trainingIndexes = 0 until 5000 by 500 flatMap (randomValues(350, _))
    val testIndexes = 0 until 5000 filter (!trainingIndexes.contains(_))
    (data._1(trainingIndexes, ::).toDenseMatrix -> data._2(trainingIndexes, ::).toDenseMatrix) ->
      (data._1(testIndexes, ::).toDenseMatrix -> data._2(testIndexes, ::).toDenseMatrix)
  }

  def readTrainingData(): (DenseMatrix[Double], DenseMatrix[Int]) = {
    readFileAsDoubles("src/main/resources/demo/X")(400, 5000).t -> readFileAsInts("src/main/resources/demo/y")(5000, 1)
  }

  private def randomValues(count: Int, floor: Int = 0, variance: Int = 499): List[Int] = {
    def generate(floor: Int, variance: Int): Stream[Int] =
      randomInt(1, (floor, floor + variance)).apply(0) #:: generate(floor, variance)

    generate(floor, variance).distinct.take(count).toList
  }

  private def readFileAsInts(path: String)(rows: Int, cols: Int): DenseMatrix[Int] = {
    val source = Source.fromFile(path)
    try {
      val elements: Iterator[Int] = (source.getLines() mkString "," split ',' map (_.toInt)).toIterator
      DenseMatrix.fill[Int](rows, cols)(elements.next)
    } finally source.close()
  }

  private def readFileAsDoubles(path: String)(rows: Int, cols: Int): DenseMatrix[Double] = {
    val source = Source.fromFile(path)
    try {
      val elements: Iterator[Double] = (source.getLines() mkString "," split ',' map (_.toDouble)).toIterator
      DenseMatrix.fill[Double](rows, cols)(elements.next)
    } finally source.close()
  }
}
