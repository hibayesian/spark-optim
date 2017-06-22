package org.apache.spark.ml.optim

import org.apache.spark.mllib.linalg.Vector

abstract class FtrlUpdater extends Serializable {
  def compute(
      weightsOld: Vector,
      gradient: Vector,
      alpha: Double,
      beta: Double,
      l1: Double,
      l2: Double,
      n: Vector,
      z: Vector): (Vector, Double, Vector, Vector)
}
