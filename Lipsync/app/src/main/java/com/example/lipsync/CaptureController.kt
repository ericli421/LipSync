package com.example.lipsync

import android.app.Activity
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Base64
import android.view.WindowManager
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

data class CaptureState(
    val active: Boolean = false,
    val status: String = "Ready to connect",
    val level: Float = 0f,
    val fps: Int = 0,
    val input: String = "Built-in microphone preferred",
    val monitoring: Boolean = false,
)

class CaptureController(private val activity: Activity, private val owner: LifecycleOwner,
                        private val update: (CaptureState) -> Unit) {
    private val main = Handler(Looper.getMainLooper())
    private val client = OkHttpClient.Builder().pingInterval(5, TimeUnit.SECONDS).build()
    private var state = CaptureState()
    @Volatile private var generation = 0
    @Volatile private var session: String? = null
    @Volatile private var socket: WebSocket? = null
    private var camera: PhoneCamera? = null
    private var audio: PhoneAudio? = null
    private val videoPending = AtomicBoolean(false)
    private val videoSeq = AtomicLong(0)
    @Volatile private var videoSentAt = 0L
    private var frames = 0
    private var lastLevel = 0L

    fun start(address: String, front: Boolean, preview: PreviewView) {
        stop("Connecting…")
        val epoch = generation
        try {
            val endpoint = CaptureProtocol.endpoint(address)
            val microphone = PhoneAudio(activity)
            audio = microphone
            val rate = microphone.prepare()
            state = CaptureState(active = true, status = "Connecting to laptop…")
            update(state)
            activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            val listener = object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    if (generation != epoch) { webSocket.cancel(); return }
                    webSocket.send(CaptureProtocol.start("${Build.MANUFACTURER} ${Build.MODEL}", rate).toString())
                }
                override fun onMessage(webSocket: WebSocket, text: String) {
                    main.post {
                        if (generation != epoch) return@post
                        try {
                            val message = JSONObject(text)
                            when (message.getString("type")) {
                                "session" -> {
                                    check(session == null) { "Unexpected session change" }
                                    session = message.getString("session_id")
                                    state = state.copy(status = "Connected · raw passthrough")
                                    update(state)
                                    microphone.start({ bytes, seq, first, time, timing, level, route ->
                                        val id = session
                                        if (generation == epoch && id != null) {
                                            send(CaptureProtocol.media("audio", id, seq, time, timing)
                                                .put("first_sample", first).put("pcm_base64", Base64.encodeToString(bytes, Base64.NO_WRAP)), epoch)
                                            val now = SystemClock.elapsedRealtime()
                                            if (now - lastLevel > 100) { lastLevel = now; main.post {
                                                if (generation == epoch) { state = state.copy(level = level, input = route); update(state) }
                                            } }
                                        }
                                    }, { fail(it, epoch) })
                                    camera = PhoneCamera(preview).also { capture ->
                                        capture.start(owner, front, { session != null && !videoPending.get() }, { jpeg, time, sensorNanos ->
                                            val id = session
                                            if (id != null && generation == epoch && videoPending.compareAndSet(false, true)) {
                                                videoSentAt = SystemClock.elapsedRealtime()
                                                send(CaptureProtocol.media("video", id, videoSeq.getAndIncrement(), time, "estimated")
                                                    .put("sensor_time_ns", sensorNanos.toString())
                                                    .put("jpeg_base64", Base64.encodeToString(jpeg, Base64.NO_WRAP)), epoch)
                                            }
                                        }, { fail(it, epoch) })
                                    }
                                }
                                "audio" -> if (message.optString("session_id") == session) microphone.enqueue(Base64.decode(message.getString("pcm_base64"), Base64.NO_WRAP))
                                "discontinuity" -> microphone.discontinuity()
                                "video_ack" -> if (message.optString("session_id") == session) {
                                    videoPending.set(false)
                                    if (message.has("error")) { state = state.copy(status = "Video: ${message.getString("error")}"); update(state) }
                                    else frames++
                                }
                                "error" -> stop(message.optString("message", "Server rejected capture"))
                            }
                        } catch (exception: Exception) { stop(exception.message ?: "Invalid server response") }
                    }
                }
                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) = fail(t.message ?: "Connection failed", epoch)
                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = fail("Disconnected. Tap Start to reconnect.", epoch)
                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) { webSocket.close(code, reason); fail("Disconnected. Tap Start to reconnect.", epoch) }
            }
            socket = client.newWebSocket(Request.Builder().url(endpoint).build(), listener)
            main.postDelayed({ if (generation == epoch && session == null) stop("Connection timed out. Check laptop address and server.") }, 10_000)
            val ticker = object : Runnable {
                override fun run() {
                    if (generation != epoch) return
                    if (videoPending.get() && videoSentAt > 0 && SystemClock.elapsedRealtime() - videoSentAt > 10_000) { stop("Video processing timed out. Reconnect to retry."); return }
                    state = state.copy(fps = frames); frames = 0; update(state)
                    main.postDelayed(this, 1000)
                }
            }
            main.postDelayed(ticker, 1000)
        } catch (exception: Exception) { stop(exception.message ?: "Unable to start capture") }
    }

    private fun send(message: JSONObject, epoch: Int) {
        val target = socket ?: return
        if (generation != epoch) return
        if (target.queueSize() > 256 * 1024 || !target.send(message.toString())) fail("Network cannot keep up. Reconnect to retry.", epoch)
    }
    private fun fail(message: String, epoch: Int) { main.post { if (generation == epoch) stop(message) } }
    fun monitor(enabled: Boolean) { audio?.monitor(enabled); state = state.copy(monitoring = enabled); update(state) }
    fun stop(message: String = "Stopped") {
        generation++; session = null
        socket?.cancel(); socket = null
        camera?.stop(); camera = null
        audio?.stop(); audio = null
        videoPending.set(false); videoSeq.set(0); frames = 0; videoSentAt = 0
        activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        state = CaptureState(status = message); update(state)
    }
    fun destroy() { stop(); client.dispatcher.executorService.shutdown(); client.connectionPool.evictAll() }
}
