package com.example.lab_4_10

import androidx.camera.lifecycle.ProcessCameraProvider
import kotlinx.coroutines.flow.StateFlow

interface CameraProvider {
    val cameraProvider: StateFlow<ProcessCameraProvider?>
}