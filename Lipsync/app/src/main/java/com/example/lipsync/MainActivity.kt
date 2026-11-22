package com.example.lipsync

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
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
import androidx.compose.ui.text.font.FontFamily
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
                                Text("Audio", style = MaterialTheme.typography.titleMedium)
                                Text("Microphone level · processed video ${captureState.fps} FPS")
                                LinearProgressIndicator(progress = { captureState.level.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                                Text(captureState.input, style = MaterialTheme.typography.bodySmall)
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("Headphone monitor", modifier = Modifier.padding(top = 12.dp))
                                    Switch(checked = captureState.monitoring, onCheckedChange = { controller.monitor(it) }, enabled = captureState.active,
                                        modifier = Modifier.semantics { contentDescription = "Headphone monitor" })
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column(Modifier.weight(1f).padding(top = 4.dp)) {
                        Text("Auto volume")
                        Text(captureState.test.volumeGainDb?.let { "Raises quiet speech: %+.0f dB now".format(it) }
                            ?: "Raises quiet speech to a comfortable level", style = MaterialTheme.typography.bodySmall)
                    }
                    Switch(checked = captureState.test.autoVolume, enabled = captureState.connected,
                        onCheckedChange = { controller.control(CaptureProtocol.layer("auto_volume", it)) },
                        modifier = Modifier.semantics { contentDescription = "Auto volume" })
                }
                Text("Use headphones to avoid feedback.", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                        SeparationTestCard(captureState, controller)
                        DebugLogCard(captureState) { text ->
                            getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("LipSync debug log", text))
                        }
                    }
                }
            }
        }
    }
    override fun onStop() { super.onStop(); if (::controller.isInitialized) controller.stop("Stopped while app is in background") }
    override fun onDestroy() { if (::controller.isInitialized) controller.destroy(); super.onDestroy() }
}

private val SOURCES = listOf("original" to "Original", "voice1" to "Voice 1", "voice2" to "Voice 2", "focus" to "Focus")
private val MODELS = listOf("auto" to "Auto", "sepformer-whamr16k" to "WHAMR 16k", "sepformer-whamr" to "WHAMR 8k",
    "resepformer-wsj02mix" to "ReSepFormer")
private val WINDOWS = listOf(1.0, 1.5, 2.0)
private val HOPS = listOf(0.25, 0.5)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SeparationTestCard(state: CaptureState, controller: CaptureController) {
    val test = state.test
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("SepFormer test (runs on laptop)", style = MaterialTheme.typography.titleMedium)
            Text(when (test.state) {
                "loading" -> "Loading model on the laptop… (first run downloads it)"
                "ready" -> "Ready · ${test.where} · waiting for audio"
                "running" -> "Running · ${test.where}"
                "error" -> "Error: ${test.error ?: "unknown"}"
                else -> "Off"
            }, color = if (test.state == "error") MaterialTheme.colorScheme.error else Color.Unspecified)
            Button(enabled = state.connected, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
                onClick = { controller.control(CaptureProtocol.testMode(!test.enabled, test.model, test.windowS, test.hopS, test.identity)) }) {
                Text(if (test.enabled) "Stop SepFormer test" else "Start SepFormer test")
            }
            fun configure(model: String = test.model, window: Double = test.windowS, hop: Double = test.hopS,
                          identity: Boolean = test.identity) =
                controller.control(CaptureProtocol.testMode(test.enabled, model, window, hop, identity))
            Text("Model (changing it reloads on the laptop)", style = MaterialTheme.typography.labelLarge)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MODELS.forEach { (value, label) ->
                    FilterChip(selected = test.model == value, enabled = state.connected,
                        onClick = { configure(model = value) }, label = { Text(label) })
                }
            }
            Text("Window / hop (delay = 0.25 s + 2 × hop)", style = MaterialTheme.typography.labelLarge)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                WINDOWS.forEach { window ->
                    FilterChip(selected = test.windowS == window, enabled = state.connected,
                        onClick = { configure(window = window) }, label = { Text("${window} s") })
                }
                HOPS.forEach { hop ->
                    FilterChip(selected = test.hopS == hop, enabled = state.connected,
                        onClick = { configure(hop = hop) }, label = { Text("hop ${hop} s") })
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("Voice lock", style = MaterialTheme.typography.labelLarge)
                    Text(when {
                        !test.identity -> "Off: voices can swap after a pause"
                        test.identityReady -> "On: keeps each person in their own voice (GPU)"
                        test.state == "loading" -> "Loading voiceprint model…"
                        else -> "On, but needs the GPU (see Debug log)"
                    }, style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = test.identity, enabled = state.connected, onCheckedChange = { configure(identity = it) },
                    modifier = Modifier.semantics { contentDescription = "Voice lock" })
            }
            Text("Person on camera (${test.facesInView} face${if (test.facesInView == 1) "" else "s"} in view, " +
                "${if (test.faceMode == "selected") "chosen" else "automatic"})", style = MaterialTheme.typography.labelLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = state.connected, onClick = { controller.control(CaptureProtocol.nextFace()) }) { Text("Next face") }
                OutlinedButton(enabled = state.connected && test.faceMode == "selected",
                    onClick = { controller.control(CaptureProtocol.autoFace()) }) { Text("Auto face") }
            }
            Text("Listen to", style = MaterialTheme.typography.labelLarge)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SOURCES.forEach { (value, label) ->
                    FilterChip(selected = test.monitor == value, enabled = state.connected,
                        onClick = { controller.control(CaptureProtocol.monitorSource(value)) }, label = { Text(label) })
                }
            }
            Text("Focus sound", style = MaterialTheme.typography.labelLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = test.focusStyle == "mask", enabled = state.connected,
                    onClick = { controller.control(CaptureProtocol.focusStyle("mask")) }, label = { Text("Natural") })
                FilterChip(selected = test.focusStyle == "mix", enabled = state.connected,
                    onClick = { controller.control(CaptureProtocol.focusStyle("mix")) }, label = { Text("Mix (separated)") })
            }
            if (!state.monitoring) Text("Turn on Headphone monitor above to hear it.", style = MaterialTheme.typography.bodySmall)
            if (test.state == "running") {
                val budget = (test.hopS * 1000).toInt()
                Text("Model ${test.inferMs?.let { "%.0f".format(it) } ?: "–"} ms (p95 ${test.p95Ms?.let { "%.0f".format(it) } ?: "–"}) / budget $budget ms")
                Text(when (test.keepingUp) {
                    true -> "Keeping up · delay ${test.delayMs ?: "–"} ms + Wi-Fi"
                    false -> "NOT keeping up · ${test.recentDrops} late hops dropped in 10 s (${test.skipped} total)"
                    null -> "Measuring…"
                }, color = if (test.keepingUp == false) MaterialTheme.colorScheme.error else Color.Unspecified)
                if (test.voiceDb.size == 2) Text("Voice 1 %.1f dB · Voice 2 %.1f dB · avg gap %s dB".format(
                    test.voiceDb[0], test.voiceDb[1], test.voiceGapDb?.let { "%.1f".format(it) } ?: "–"))
                Text("Model audio ${test.modelRate?.let { "${it / 1000} kHz" } ?: "–"} · Wi-Fi max gap ${test.maxArrivalGapMs ?: "–"} ms (5 s)")
                Text("On camera: ${if (test.pick >= 0) "Voice ${test.pick + 1}" else "none yet"} (${test.pickReason}" +
                    "${test.lipEvidence?.let { ", evidence %+.2f".format(it) } ?: ""}, ${test.pickSwitches} switches)")
                test.identityDecision?.let { Text("Voice lock: ${test.identityCorrections} corrections · $it", style = MaterialTheme.typography.bodySmall) }
                test.liveError?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        }
    }
}

@Composable
private fun DebugLogCard(state: CaptureState, copy: (String) -> Unit) {
    val laptop = state.serverLog.asReversed()
    val phone = state.phoneLog.asReversed()
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Debug log", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 12.dp))
                OutlinedButton(onClick = {
                    copy((listOf("Laptop:") + laptop.map { it.text() } + listOf("", "Phone:") + phone).joinToString("\n"))
                }) { Text("Copy") }
            }
            Column(Modifier.fillMaxWidth().heightIn(max = 320.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("Laptop (newest first)", style = MaterialTheme.typography.labelLarge)
                if (laptop.isEmpty()) Text("No laptop messages yet. Start capture to connect.", style = MaterialTheme.typography.bodySmall)
                laptop.forEach { line ->
                    Text(line.text(), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                        color = when (line.level) {
                            "error" -> MaterialTheme.colorScheme.error
                            "warn" -> MaterialTheme.colorScheme.tertiary
                            else -> Color.Unspecified
                        })
                }
                Spacer(Modifier.height(8.dp))
                Text("Phone (newest first)", style = MaterialTheme.typography.labelLarge)
                phone.forEach { Text(it, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall) }
            }
        }
    }
}
