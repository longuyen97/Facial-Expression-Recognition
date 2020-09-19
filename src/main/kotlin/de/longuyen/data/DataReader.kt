package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray

interface DataReader {
    fun features() : INDArray

    fun targets() : INDArray
}