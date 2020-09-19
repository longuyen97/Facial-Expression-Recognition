package de.longuyen

import de.longuyen.data.BinaryClassificationDataReader
import de.longuyen.svm.SVM
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.indexing.NDArrayIndex
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

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
        var correct = 0.0
        var wrong = 0.0
        for (j in yPrediction.indices) {
            val prediction = yPrediction[i]
            val truth = yTest[i]
            if (prediction != truth) {
                wrong += 1.0
            } else {
                correct += 1.0
            }
        }
    }
    if(!File("target/prediction-happy").exists()){
        File("target/prediction-happy").mkdir()
    }
    if(!File("target/prediction-surprised").exists()){
        File("target/prediction-surprised").mkdir()
    }
    val yPrediction = svm.predict(xTest).castTo(DataType.INT32).toIntVector()
    val images = xTest.mul(255f).castTo(DataType.INT32)
    for (i in yPrediction.indices) {
        val image = images.get(NDArrayIndex.interval(i, i + 1)).reshape(intArrayOf(48, 48)).transpose().toIntMatrix()
        val bufferedImage = BufferedImage(48, 48, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until 48) {
            for (x in 0 until 48) {
                bufferedImage.setRGB(x, y, Color(image[y][x], image[y][x], image[y][x]).rgb)
            }
        }
        if (yPrediction[i] == 1) {
            ImageIO.write(bufferedImage, "png", File("target/prediction-happy/$i.png"))
        } else {
            ImageIO.write(bufferedImage, "png", File("target/prediction-surprised/$i.png"))
        }
    }
}