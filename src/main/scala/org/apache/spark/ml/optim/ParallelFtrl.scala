package org.apache.spark.ml.optim

import breeze.linalg.norm
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * :: DeveloperApi ::
  * Class used to solve an online optimization problem using Follow-the-regularized-leader.
  * It can give a good performance vs. sparsity tradeoff.
  *
  * Reference: [Ad Click Prediction: a View from the Trenches](https://www.eecs.tufts.
  * edu/~dsculley/papers/ad-click-prediction.pdf)
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
class ParallelFtrl private[spark](private var gradient: FtrlGradient, private var updater: FtrlUpdater)
  extends Optimizer with Logging {

  private var alpha: Double = 0.01
  private var beta: Double = 1.0
  private var l1: Double = 0.1
  private var l2: Double = 1.0
  private var numIterations: Int = 1
  private var convergenceTol: Double = 0.001
  private var aggregationDepth: Int = 2
  private var numPartitions: Int = -1

  /**
    * Set the alpha. Default 0.01.
    */
  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  /**
    * Set the beta. Default 1.0.
    */
  def setBeta(beta: Double): this.type = {
    this.beta = beta
    this
  }

  /**
    * Set the l1. Default 0.1.
    */
  def setL1(l1: Double): this.type = {
    this.l1 = l1
    this
  }

  /**
    * Set the l2. Default 1.0.
    */
  def setL2(l2: Double): this.type = {
    this.l2 = l2
    this
  }

  /**
    * Set the number of iterations for parallel Ftrl. Default 1.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001.
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    * - If the norm of the new solution vector is >1, the diff of solution vectors
    * is compared to relative tolerance which means normalizing by the norm of
    * the new solution vector.
    * - If the norm of the new solution vector is <=1, the diff of solution vectors
    * is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the aggregation depth. Default 2.
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    */
  def setAggregationDepth(aggregationDepth: Int): this.type = {
    this.aggregationDepth = aggregationDepth
    this
  }

  /**
    * Set the number of partitions for parallel SGD.
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    this.numPartitions = numPartitions
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for parallel Ftrl.
    */
  def setGradient(gradient: FtrlGradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: FtrlUpdater): this.type = {
    this.updater = updater
    this
  }

  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = ParallelFtrl.runFtrl(
      data,
      gradient,
      updater,
      alpha,
      beta,
      l1,
      l2,
      initialWeights,
      numIterations,
      convergenceTol,
      aggregationDepth,
      numPartitions)
    weights
  }
}

object ParallelFtrl extends Logging {
  def runFtrl(
      data: RDD[(Double, Vector)],
      gradient: FtrlGradient,
      updater: FtrlUpdater,
      alpha: Double,
      beta: Double,
      l1: Double,
      l2: Double,
      initialWeights: Vector,
      numIterations: Int,
      convergenceTol: Double,
      aggregationDepth: Int,
      numPartitions: Int): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("Ftrl.runFtrl returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)

    val numParts = if (numPartitions > 0) numPartitions else data.getNumPartitions

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      val (avgWeights, avgRegVal, lossSum, batchSize) = data.repartition(numParts)
        .mapPartitions { part =>
          var localWeights = bcWeights.value
          var localRegVal = 0.0
          var localLossSum = 0.0
          var n = Vectors.zeros(localWeights.size)
          var z = Vectors.zeros(localWeights.size)
          var j = 1
          while (part.hasNext) {
            val (label, vector) = part.next()
            val (localGrad, localLoss) = gradient.compute(vector, label, localWeights)
            val update = updater.compute(localWeights, localGrad, alpha, beta, l1, l2, n, z)
            localWeights = update._1
            localRegVal = update._2
            n = update._3
            z = update._4
            localLossSum += localLoss
            j += 1
          }
          Iterator.single((localWeights, localRegVal, localLossSum, j))
        }.treeReduce ({ case ((w1, rv1, ls1, c1), (w2, rv2, ls2, c2)) =>
        val avgWeights =
          (w1.asBreeze * c1.toDouble + w2.asBreeze * c2.toDouble) / (c1 + c2).toDouble
        val avgRegVal = (rv1 * c1.toDouble + rv2 * c2.toDouble) / (c1 + c2).toDouble
        (Vectors.fromBreeze(avgWeights), avgRegVal, ls1 + ls2, c1 + c2)}, aggregationDepth)
      stochasticLossHistory.append(lossSum / batchSize + avgRegVal)
      weights = avgWeights
      previousWeights = currentWeights
      currentWeights = Some(weights)
      if (previousWeights.isDefined && currentWeights.isDefined) {
        converged = isConverged(previousWeights.get, currentWeights.get, convergenceTol)
      }
      i += 1
    }

    logInfo("ParallelFtrl.runParallelFtrl finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)
  }

  private def isConverged(
      previousWeights: Vector,
      currentWeights: Vector,
      convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
}
