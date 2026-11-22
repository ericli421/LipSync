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

    /** Start/stop the laptop's live separation test (same control the dashboard sends). */
    fun testMode(enabled: Boolean, model: String = "auto", windowS: Double = 1.5, hopS: Double = 0.25) = JSONObject()
        .put("type", "set_test_mode").put("enabled", enabled).put("model", model)
        .put("window_s", windowS).put("hop_s", hopS)

    /** What the laptop sends back for monitoring: original, voice1, voice2 or focus. */
    fun monitorSource(source: String) = JSONObject().put("type", "set_monitor").put("source", source)
}

/** The laptop's separation test, as reported in its `pipeline` status messages. */
data class TestStatus(
    val enabled: Boolean = false,
    val state: String = "off",
    val where: String = "",
    val error: String? = null,
    val monitor: String = "original",
    val windowS: Double = 1.5,
    val hopS: Double = 0.25,
    val inferMs: Double? = null,
    val p95Ms: Double? = null,
    val keepingUp: Boolean? = null,
    val hops: Int = 0,
    val skipped: Int = 0,
    val delayMs: Int? = null,
    val voiceDb: List<Double> = emptyList(),
    val pick: Int = -1,
    val pickReason: String = "",
    val liveError: String? = null,
)

data class LogLine(val seq: Long, val time: String, val level: String, val source: String, val message: String) {
    fun text() = "$time ${level.uppercase()} [$source] $message"
}

object PipelineStatus {
    private fun JSONObject.double(key: String): Double? = if (has(key) && !isNull(key)) getDouble(key) else null
    private fun JSONObject.text(key: String): String? = if (has(key) && !isNull(key)) getString(key) else null

    fun parse(message: JSONObject): Pair<TestStatus, List<LogLine>> {
        val log = message.optJSONArray("log")?.let { lines ->
            (0 until lines.length()).map { lines.getJSONObject(it) }.map {
                LogLine(it.optLong("seq"), it.optString("time"), it.optString("level"), it.optString("source"), it.optString("message"))
            }
        } ?: emptyList()
        val test = message.optJSONObject("test") ?: return TestStatus() to log
        val info = test.optJSONObject("info")
        val live = test.optJSONObject("live")
        val where = info?.let { listOfNotNull(it.text("model"), it.text("gpu") ?: it.text("device")).joinToString(" on ") } ?: ""
        val status = TestStatus(
            enabled = test.optBoolean("enabled"), state = test.optString("state", "off"), where = where,
            error = test.text("error"), monitor = test.optString("monitor", "original"),
            windowS = test.optDouble("window_s", 1.5), hopS = test.optDouble("hop_s", 0.25),
            inferMs = live?.double("infer_ms"), p95Ms = live?.double("infer_p95_ms"),
            keepingUp = live?.takeIf { it.has("keeping_up") && !it.isNull("keeping_up") }?.getBoolean("keeping_up"),
            hops = live?.optInt("hops") ?: 0, skipped = live?.optInt("skipped_hops") ?: 0,
            delayMs = live?.takeIf { it.has("delay_ms") }?.getInt("delay_ms"),
            voiceDb = live?.optJSONArray("voice_db")?.let { db -> (0 until db.length()).map { db.getDouble(it) } } ?: emptyList(),
            pick = live?.optInt("pick", -1) ?: -1, pickReason = live?.optString("pick_reason") ?: "",
            liveError = live?.text("error"),
        )
        return status to log
    }
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
