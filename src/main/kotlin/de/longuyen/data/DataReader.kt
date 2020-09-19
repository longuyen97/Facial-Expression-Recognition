package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.awt.Color
import java.awt.image.BufferedImage

interface DataReader {
    fun features(): INDArray

    fun targets(): INDArray

    fun shuffle()

    fun img2array(bufferedImage: BufferedImage): Array<Float> {
        val pixels = mutableListOf<Float>()
        for (y in 0 until bufferedImage.height) {
            for (x in 0 until bufferedImage.width) {
                val color = Color(bufferedImage.getRGB(y, x))
                pixels.add((color.red + color.green + color.blue).toFloat() / 3f)
            }
        }
        return pixels.toTypedArray()
    }
}