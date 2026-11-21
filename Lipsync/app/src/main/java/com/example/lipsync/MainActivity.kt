package com.example.lipsync

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.lipsync.ui.theme.LipsyncTheme

class MainActivity : ComponentActivity() {
    private var captureState by mutableStateOf(CaptureState())
    private lateinit var controller: CaptureController
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        controller = CaptureController(this, this) { captureState = it }
        val preferences = getSharedPreferences("capture", MODE_PRIVATE)
        setContent {
            LipsyncTheme {
                var address by rememberSaveable { mutableStateOf(preferences.getString("address", "") ?: "") }
                var front by rememberSaveable { mutableStateOf(false) }
                var permissionError by remember { mutableStateOf(false) }
                val preview = remember { PreviewView(this).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE } }
                fun start() {
                    preferences.edit().putString("address", address.trim()).apply()
                    controller.start(address, front, preview)
                }
                val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
                    permissionError = result.values.any { !it }
                    if (!permissionError) start()
                }
                Scaffold { padding ->
                    Column(Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        Text("LipSync", style = MaterialTheme.typography.headlineLarge)
                        Text("Local capture bridge", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        OutlinedTextField(value = address, onValueChange = { address = it }, label = { Text("Laptop address") },
                            placeholder = { Text("192.168.1.10:8000") }, supportingText = { Text("Use the laptop’s LAN address on the same Wi-Fi.") },
                            singleLine = true, enabled = !captureState.active, modifier = Modifier.fillMaxWidth())
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Button(modifier = Modifier.weight(1f).heightIn(min = 48.dp), enabled = captureState.active || address.isNotBlank(), onClick = {
                                if (captureState.active) controller.stop() else {
                                    val needed = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                                    if (needed.all { ContextCompat.checkSelfPermission(this@MainActivity, it) == PackageManager.PERMISSION_GRANTED }) {
                                        permissionError = false; start()
                                    } else permissions.launch(needed)
                                }
                            }) { Text(if (captureState.active) "Stop capture" else "Start capture") }
                            OutlinedButton(enabled = !captureState.active, modifier = Modifier.heightIn(min = 48.dp), onClick = { front = !front }) {
                                Text(if (front) "Front camera" else "Rear camera")
                            }
                        }
                        if (permissionError) Text("Camera and microphone permissions are required. Allow them in app settings if Android no longer shows a prompt.", color = MaterialTheme.colorScheme.error)
                        Text(captureState.status, style = MaterialTheme.typography.bodyLarge)
                        AndroidView(factory = { preview }, modifier = Modifier.fillMaxWidth().aspectRatio(4f / 3f).background(Color.Black))
                        Card(Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                Text("Raw passthrough", style = MaterialTheme.typography.titleMedium)
                                Text("Microphone level · processed video ${captureState.fps} FPS")
                                LinearProgressIndicator(progress = { captureState.level.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                                Text(captureState.input, style = MaterialTheme.typography.bodySmall)
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("Headphone monitor", modifier = Modifier.padding(top = 12.dp))
                                    Switch(checked = captureState.monitoring, onCheckedChange = { controller.monitor(it) }, enabled = captureState.active,
                                        modifier = Modifier.semantics { contentDescription = "Headphone monitor" })
                                }
                                Text("Use headphones to avoid feedback. Voice isolation and captions are not enabled yet.", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            }
        }
    }
    override fun onStop() { super.onStop(); if (::controller.isInitialized) controller.stop("Stopped while app is in background") }
    override fun onDestroy() { if (::controller.isInitialized) controller.destroy(); super.onDestroy() }
}
