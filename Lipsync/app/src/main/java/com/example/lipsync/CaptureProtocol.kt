package com.example.lipsync

import java.net.URI
import org.json.JSONObject

object CaptureProtocol {
    fun endpoint(address: String): String {
        val input = address.trim().let { if (it.contains("://")) it else "ws://$it" }
        val uri = URI(input)
        val scheme = when (uri.scheme) {
            "http", "ws" -> "ws"
            "https", "wss" -> "wss"
            else -> throw IllegalArgumentException("Use a laptop address such as 192.168.1.10:8000")
        }
        require(!uri.host.isNullOrBlank() && uri.userInfo == null) { "Enter a valid laptop address" }
        return URI(scheme, null, uri.host, uri.port, "/ws/capture", null, null).toString()
    }

    fun start(device: String, rate: Int) = JSONObject()
        .put("type", "start").put("protocol_version", 1).put("device_name", device.take(100))
        .put("sample_rate", rate).put("channels", 1).put("format", "pcm_s16le")

    fun media(type: String, session: String, seq: Long, captureUs: Long, timing: String) = JSONObject()
        .put("type", type).put("session_id", session).put("seq", seq)
        .put("capture_time_us", captureUs).put("timing", timing)
}

class AudioSampleClock(val rate: Int) {
    var nextSample = 0L
        private set
    var sequence = 0L
        private set
    val blockSamples get() = rate / 50
    fun advance(): Pair<Long, Long> {
        val result = sequence to nextSample
        sequence++
        nextSample += blockSamples
        return result
    }
    fun timeUs(firstSample: Long, anchorFrame: Long, anchorNanos: Long): Long =
        anchorNanos / 1000 + (firstSample - anchorFrame) * 1_000_000L / rate
}
