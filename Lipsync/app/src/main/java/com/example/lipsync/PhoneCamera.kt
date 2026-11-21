package com.example.lipsync

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import android.util.Size
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class PhoneCamera(private val previewView: PreviewView) {
    private val executor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null
    @Volatile private var stopped = false

    fun start(owner: LifecycleOwner, front: Boolean, canSend: () -> Boolean,
              frame: (ByteArray, Long, Long) -> Unit, error: (String) -> Unit) {
        val future = ProcessCameraProvider.getInstance(previewView.context)
        future.addListener({
            if (stopped) return@addListener
            try {
                provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                        ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                var lastSent = 0L
                analysis.setAnalyzer(executor) { image ->
                    try {
                        val now = SystemClock.elapsedRealtimeNanos()
                        if (!stopped && now - lastSent >= 40_000_000L && canSend()) {
                            lastSent = now
                            val original = image.toBitmap()
                            val scale = minOf(1f, 640f / maxOf(original.width, original.height))
                            val matrix = Matrix().apply { postScale(scale, scale); postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                            val upright = Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
                            val bytes = ByteArrayOutputStream().use { output ->
                                upright.compress(Bitmap.CompressFormat.JPEG, 75, output); output.toByteArray()
                            }
                            frame(bytes, now / 1000, image.imageInfo.timestamp)
                            if (upright !== original) upright.recycle()
                            original.recycle()
                        }
                    } catch (exception: Exception) { if (!stopped) error(exception.message ?: "Camera capture failed") }
                    finally { image.close() }
                }
                val selector = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                provider!!.bindToLifecycle(owner, selector, preview, analysis)
            } catch (exception: Exception) { error(exception.message ?: "Camera unavailable") }
        }, ContextCompat.getMainExecutor(previewView.context))
    }

    fun stop() { stopped = true; provider?.unbindAll(); executor.shutdown() }
}
