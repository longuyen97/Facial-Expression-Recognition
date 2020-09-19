package de.longuyen

import de.longuyen.data.BinaryClassificationDataReader
import de.longuyen.svm.SVM
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.indexing.NDArrayIndex

fun main() {
    val reader = BinaryClassificationDataReader()
    reader.shuffle()
    val x = reader.features().div(255f)
    val y = reader.targets()

    val xTest = x.get(NDArrayIndex.interval(0, 300), NDArrayIndex.interval(0, 2304))
    val yTest = y.get(NDArrayIndex.interval(0, 300)).castTo(DataType.INT32).toIntVector()
    val xTrain = x.get(NDArrayIndex.interval(300, 414), NDArrayIndex.interval(0, 2304))
    val yTrain = y.get(NDArrayIndex.interval(300, 414))

    val svm = SVM()
    for (i in 0 until 100) {
        svm.train(xTrain, yTrain)
        val yPrediction = svm.predict(xTest).castTo(DataType.INT32).toIntVector()
        var accuracy = 0.0
        for (j in yPrediction.indices) {
            if (yPrediction[i] == yTest[i]) {
                accuracy += 1.0
            }
        }
        println(accuracy / yPrediction.size.toDouble())
    }
}