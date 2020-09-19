package de.longuyen.data

import kotlin.test.Test
import kotlin.test.assertEquals

class TestBinaryClassificationDataReader {
    @Test
    fun testReadData() {
        val reader = BinaryClassificationDataReader()
        val x1 = reader.features()
        val y1 = reader.targets()
        assertEquals(x1.shape()[0], 414)
        assertEquals(x1.shape()[1], 2304)
        assertEquals(y1.shape()[0], 414)
    }
}