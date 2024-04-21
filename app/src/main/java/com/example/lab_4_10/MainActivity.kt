package com.example.lab_4_10

import android.Manifest
import android.os.Bundle
import android.util.Log
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.lab_4_10.databinding.ActivityMainBinding
import com.example.lab_4_10.ml.Yolo
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {
    private val activityResultLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ){ permissionGranted ->
        if (permissionGranted) {
            startCamera()
        }
    }

    private val imageAnalyzer: ImageAnalyze by lazy{
        ImageAnalyze(model, imageProcessor, labels) { detectedObjects ->
            Log.e(
                "class",
                "${detectedObjects}"
            )
        }
    }

    private lateinit var model: Yolo
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var labels: List<String>

    private lateinit var bindding: ActivityMainBinding


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        bindding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(bindding.root)

        model = Yolo.newInstance(this)
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        labels = FileUtil.loadLabels(this, "label.txt")

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        activityResultLauncher.launch(Manifest.permission.CAMERA)

    }

    override fun onDestroy() {
        super.onDestroy()
        closeCamera()
    }

    private fun startCamera() {
        val center = CameraRecognitionCenter(applicationContext)
        center.setupCamera(this)
        lifecycleScope.launch {
            center.cameraProvider
                .filterNotNull()
                .collectLatest {
                    val preview = Preview.Builder().build()
                    it.bindToLifecycle(
                        this@MainActivity,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview
                    )
                    preview.setSurfaceProvider(bindding.viewFinder.surfaceProvider)
                }
        }
    }

    private fun closeCamera(){
        model.close()
    }
}