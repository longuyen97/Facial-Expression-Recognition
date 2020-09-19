package de.longuyen

import de.longuyen.data.BinaryClassificationDataReader
import de.longuyen.svm.SVM
import org.nd4j.evaluation.classification.EvaluationBinary
import org.nd4j.linalg.indexing.NDArrayIndex

fun main() {
    val reader = BinaryClassificationDataReader()
    reader.shuffle()
    val x = reader.features().div(255f)
    val y = reader.targets()

    val xTest = x.get(NDArrayIndex.interval(0, 300), NDArrayIndex.interval(0, 2304))
    val yTest = y.get(NDArrayIndex.interval(0, 300))
    val xTrain = x.get(NDArrayIndex.interval(300, 414), NDArrayIndex.interval(0, 2304))
    val yTrain = y.get(NDArrayIndex.interval(300, 414))

    val svm = SVM(xTrain, yTrain)
    svm.train()
    val yPrediction = svm.predict(xTest)


}