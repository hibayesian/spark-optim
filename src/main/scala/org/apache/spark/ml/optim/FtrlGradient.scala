package org.apache.spark.ml.optim

import org.apache.spark.mllib.linalg.Vector

abstract class FtrlGradient extends Serializable {
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double)
}
