package de.longuyen.data

import org.apache.log4j.LogManager
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.max
import kotlin.math.min

/**
 * Return the data of two expressions Happy and Surprise
 */
class BinaryClassificationDataReader : DataReader {
    private val log = LogManager.getLogger(BinaryClassificationDataReader::class.java)
    private val features = mutableListOf<File>()
    private val targets = mutableListOf<Int>()

    init {
        val happy = File("data/happy")
        val surprise = File("data/surprise")
        val happyFiles = happy.listFiles()
        val surpriseFiles = surprise.listFiles()
        log.debug("Happy contains ${happyFiles!!.size} files. Surprise contains ${surpriseFiles!!.size} files.")
        val dataPoints = min(happyFiles.size, surpriseFiles.size)
        for(i in 0 until dataPoints){
            features.add(happyFiles[i])
            targets.add(1)
            features.add(surpriseFiles[i])
            targets.add(-1)
        }
    }

    override fun features(): INDArray {
        val dataPoints = mutableListOf<Array<Float>>()
        for(feature in features){
            val image = ImageIO.read(feature)
            val imageArray = this.img2array(image)
            dataPoints.add(imageArray)
        }
        return Nd4j.createFromArray(dataPoints.toTypedArray())
    }

    override fun targets(): INDArray {
        return Nd4j.createFromArray(targets.toTypedArray())
    }

    override fun shuffle() {
        val indices = mutableListOf<Int>()
        for(i in 0 until features.size){
            indices.add(i)
        }
        indices.shuffle()
        for(i in indices.indices){
            val swapIndex = indices[i]

            val tempFeature = features[i]
            features[i] = features[swapIndex]
            features[swapIndex] = tempFeature

            val tempTarget = targets[i]
            targets[i] = targets[swapIndex]
            targets[swapIndex] = tempTarget
        }
    }
}