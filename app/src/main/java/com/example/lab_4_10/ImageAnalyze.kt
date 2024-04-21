package com.example.lab_4_10

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.lab_4_10.ml.Yolo
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage

class ImageAnalyze(
    private val model: Yolo,
    private val imageProcessor: ImageProcessor,
    private val labels: List<String>,
    private val onResult: (List<DetectedObject>) -> Unit
): ImageAnalysis.Analyzer{

    override fun analyze(image: ImageProxy) {
        val bitmap: Bitmap = imageProxyToBitmap(image)

        var tensorImage = TensorImage.fromBitmap(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputs = model.process(tensorImage)

        // Обработка результатов
        val detectedObjects = processOutputs(outputs, bitmap)

        // Передача результатов через callback
        onResult(detectedObjects)

        image.close()
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun processOutputs(outputs: Yolo.Outputs, bitmap: Bitmap): List<DetectedObject> {
        // Здесь можете выполнить обработку выходных данных модели YOLO и создать список обнаруженных объектов
        // Например:
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val h = mutable.height
        val w = mutable.width

        // Создаем список обнаруженных объектов
        var x: Int
        val detectedObjects = mutableListOf<DetectedObject>()
        scores.forEachIndexed { index, fl ->
            x = index * 4

            val objectLocation = ObjectLocation(
                locations.get(x+1)*w,
                locations.get(x)*h,
                locations.get(x+3)*w,
                locations.get(x+2)*h
            )
            val detectedObject = DetectedObject(
                objectLocation,
                labels.get(classes.get(index).toInt()),
                fl
            )
            detectedObjects.add(detectedObject)
        }

        return detectedObjects
    }

    data class DetectedObject(val locations: ObjectLocation, val objectClass: String, val scores: Float)
    data class ObjectLocation(val left: Float, val top: Float, val right: Float, val bottom: Float)
}