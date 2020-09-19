package de.longuyen.svm

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms
import java.lang.Exception

class SVM {
    private var weights: INDArray
    private var initialized = false

    init {
        weights = Nd4j.zeros(1)
    }

    fun predict(x: INDArray): INDArray {
        if (!initialized) {
            weights = Nd4j.randn(x.shape()[1])
            initialized = true
        }
        return Transforms.sign(weights.reshape(intArrayOf(1, weights.shape()[0].toInt())).mmul(x.transpose()))
    }

    private fun gradients(W: INDArray, X: INDArray, Y: INDArray, slack: Float): INDArray {
        val distances = Y.mul(X.mmul(W))
        val distanceVector = (Nd4j.onesLike(distances).sub(distances)).toFloatVector()

        var dw = Nd4j.zerosLike(W)

        for (i in distanceVector.indices) {
            val distance = distanceVector[i]
            if (distance < 0) {
                dw = dw.add(W)
            } else {
                val yi = Y.get(NDArrayIndex.interval(i, i + 1))
                val xi = X.get(NDArrayIndex.interval(i, i + 1), NDArrayIndex.interval(0, X.shape()[1]))
                val penalty = yi.mul(xi).mul(slack)
                dw = dw.add(W.sub(penalty))
            }
        }

        dw = dw.sub(Y.shape()[0])
        return try {
            dw.reshape(intArrayOf(dw.shape()[1].toInt()))
        }catch (e: Exception){
            dw
        }
    }

    fun train(features: INDArray, targets: INDArray, learningRate: Float = 0.000001f, slack: Float = 10000f) {
        if (!initialized) {
            weights = Nd4j.randn(features.shape()[1])
            initialized = true
        }
        for(i in 0 until features.shape()[0]) {
            val yi = targets.get(NDArrayIndex.interval(i, i + 1))
            val xi = features.get(NDArrayIndex.interval(i, i + 1), NDArrayIndex.interval(0, features.shape()[1]))
            val dW = gradients(weights, xi, yi, slack).mul(learningRate)
            weights = weights.sub(dW)
        }
    }
}