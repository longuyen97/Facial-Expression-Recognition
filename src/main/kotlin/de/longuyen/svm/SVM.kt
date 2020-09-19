package de.longuyen.svm

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

private fun gradients(W: INDArray, X: INDArray, Y: INDArray, C: Float = 10f): INDArray {
    val distances = Y.mul(X.mmul(W)).mul(-1f).add(1f)
    val distanceVector = distances.toFloatVector()
    var dw = Nd4j.zeros(W.shape()[0])
    for (i in distanceVector.indices) {
        val d = distanceVector[i]
        if (d < 0) {
            dw = dw.add(W)
        } else {
            val p = Y.get(NDArrayIndex.interval(i, i + 1))
                .mul(X.get(NDArrayIndex.interval(i, i + 1), NDArrayIndex.interval(0, X.shape()[1])))
                .mul(C)
            dw = dw.add(W.sub(p))
        }
    }
    return dw.sub(Y.shape()[0]).reshape(intArrayOf(dw.shape()[1].toInt()))
}

class SVM {
    private var weights: INDArray
    private var initialized = false

    init {
        weights = Nd4j.zeros(1)
    }

    fun predict(x: INDArray): INDArray {
        return Transforms.sign(weights.reshape(intArrayOf(1, weights.shape()[0].toInt())).mmul(x.transpose()))
    }

    fun train(features: INDArray, targets: INDArray, learningRate: Float = 0.000001f) {
        if (!initialized) {
            weights = Nd4j.randn(features.shape()[1])
            initialized = true
        }
        val dW = gradients(weights, features, targets)
        weights = weights.sub(dW.mul(learningRate))
    }
}