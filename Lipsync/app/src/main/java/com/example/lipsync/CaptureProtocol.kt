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
    fun testMode(enabled: Boolean, model: String = "auto", windowS: Double = 2.0, hopS: Double = 0.25,
                 identity: Boolean = false) = JSONObject()
        .put("type", "set_test_mode").put("enabled", enabled).put("model", model)
        .put("window_s", windowS).put("hop_s", hopS).put("identity", identity)

    /** Turn a laptop audio layer on or off: voice_band or auto_volume. */
    fun layer(name: String, enabled: Boolean) = JSONObject().put("type", "set_layer").put("layer", name).put("enabled", enabled)

    /** Focus style: "mask" keeps the original sound and lowers the others; "mix" uses the separated voice. */
    fun focusStyle(style: String) = JSONObject().put("type", "set_focus").put("style", style)

    /** Choose whose lips drive Focus: the next face left to right, or automatic. */
    fun nextFace() = JSONObject().put("type", "select_face").put("next", true)
    fun autoFace() = JSONObject().put("type", "select_face").put("clear", true)

    /** What the laptop sends back for monitoring: original, voice1, voice2 or focus. */
    fun monitorSource(source: String) = JSONObject().put("type", "set_monitor").put("source", source)
}

/** The laptop's separation test, as reported in its `pipeline` status messages. */
data class TestStatus(
    val enabled: Boolean = false,
    val state: String = "off",
    val model: String = "auto",
    val where: String = "",
    val error: String? = null,
    val monitor: String = "focus",
    val focusStyle: String = "mask",
    val lipEvidence: Double? = null,
    val pickSwitches: Int = 0,
    val windowS: Double = 2.0,
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
    val recentDrops: Int = 0,
    val voiceGapDb: Double? = null,
    val modelRate: Int? = null,
    val maxArrivalGapMs: Int? = null,
    val identity: Boolean = false,
    val autoVolume: Boolean = true,
    val faceMode: String = "auto",
    val facesInView: Int = 0,
    val volumeGainDb: Double? = null,
    val identityReady: Boolean = false,
    val identityDecision: String? = null,
    val identityCorrections: Int = 0,
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
        val layers = message.optJSONObject("layers")
        val autoVolume = layers?.optBoolean("auto_volume", true) ?: true
        val volumeGain = message.double("volume_gain_db")
        val face = message.optJSONObject("face")
        val faceMode = face?.optString("mode", "auto") ?: "auto"
        val faces = face?.optInt("faces") ?: 0
        val test = message.optJSONObject("test")
            ?: return TestStatus(autoVolume = autoVolume, volumeGainDb = volumeGain, faceMode = faceMode, facesInView = faces) to log
        val network = message.optJSONObject("network")
        val lock = test.optJSONObject("live")?.optJSONObject("identity")
        val info = test.optJSONObject("info")
        val live = test.optJSONObject("live")
        val where = info?.let { listOfNotNull(it.text("model"), it.text("gpu") ?: it.text("device")).joinToString(" on ") } ?: ""
        val status = TestStatus(
            enabled = test.optBoolean("enabled"), state = test.optString("state", "off"), where = where,
            model = test.optString("model", "auto"),
            error = test.text("error"), monitor = test.optString("monitor", "focus"),
            focusStyle = test.optString("focus_style", "mask"),
            lipEvidence = live?.double("lip_evidence"), pickSwitches = live?.optInt("pick_switches") ?: 0,
            windowS = test.optDouble("window_s", 2.0), hopS = test.optDouble("hop_s", 0.25),
            inferMs = live?.double("infer_ms"), p95Ms = live?.double("infer_p95_ms"),
            keepingUp = live?.takeIf { it.has("keeping_up") && !it.isNull("keeping_up") }?.getBoolean("keeping_up"),
            hops = live?.optInt("hops") ?: 0, skipped = live?.optInt("skipped_hops") ?: 0,
            delayMs = live?.takeIf { it.has("delay_ms") }?.getInt("delay_ms"),
            voiceDb = live?.optJSONArray("voice_db")?.let { db -> (0 until db.length()).map { db.getDouble(it) } } ?: emptyList(),
            pick = live?.optInt("pick", -1) ?: -1, pickReason = live?.optString("pick_reason") ?: "",
            liveError = live?.text("error"),
            recentDrops = live?.optInt("recent_drops") ?: 0,
            voiceGapDb = live?.double("voice_gap_db"),
            modelRate = live?.takeIf { it.has("model_rate") }?.getInt("model_rate"),
            maxArrivalGapMs = network?.takeIf { it.has("max_gap_ms") }?.getInt("max_gap_ms"),
            identity = test.optBoolean("identity", false),
            autoVolume = autoVolume, volumeGainDb = volumeGain, faceMode = faceMode, facesInView = faces,
            identityReady = test.optBoolean("identity_ready"),
            identityDecision = lock?.text("decision"),
            identityCorrections = lock?.optInt("identity_swaps") ?: 0,
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
