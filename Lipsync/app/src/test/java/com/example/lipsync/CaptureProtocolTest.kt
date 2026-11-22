package com.example.lipsync

import org.junit.Assert.*
import org.junit.Test

class CaptureProtocolTest {
    @Test fun addressesUseCaptureEndpoint() {
        assertEquals("ws://192.168.1.10:8000/ws/capture", CaptureProtocol.endpoint("192.168.1.10:8000"))
        assertEquals("ws://10.0.2.2:8000/ws/capture", CaptureProtocol.endpoint("http://10.0.2.2:8000/ws"))
        assertEquals("wss://example.com/ws/capture", CaptureProtocol.endpoint("https://example.com/"))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsCredentials() { CaptureProtocol.endpoint("ws://user:password@host") }
    @Test fun sampleClockPreservesOffsetsForEveryRate() {
        for (rate in listOf(16000, 44100, 48000)) {
            val clock = AudioSampleClock(rate)
            repeat(50) { seq -> assertEquals(seq.toLong() to seq.toLong() * rate / 50, clock.advance()) }
            assertEquals(rate.toLong(), clock.nextSample)
            assertEquals(2_000_000L, clock.timeUs(rate.toLong(), 0, 1_000_000_000L))
        }
    }
    @Test fun testControlsMatchServerProtocol() {
        val start = CaptureProtocol.testMode(true)
        assertEquals("set_test_mode", start.getString("type"))
        assertTrue(start.getBoolean("enabled"))
        assertEquals("auto", start.getString("model"))
        assertEquals(1.5, start.getDouble("window_s"), 0.0)
        assertEquals(0.25, start.getDouble("hop_s"), 0.0)
        assertEquals("focus", CaptureProtocol.monitorSource("focus").getString("source"))
    }
    @Test fun parsesPipelineStatusWithNullsAndLog() {
        val idle = org.json.JSONObject("""{"type":"pipeline","test":{"enabled":false,"state":"off","error":null,
            "info":null,"monitor":"original","window_s":1.5,"hop_s":0.25,"live":null},
            "log":[{"seq":3,"time":"20:01:02","level":"warn","source":"test","message":"model too slow"}]}""")
        val (test, log) = PipelineStatus.parse(idle)
        assertEquals("off", test.state)
        assertNull(test.inferMs)
        assertNull(test.keepingUp)
        assertEquals("20:01:02 WARN [test] model too slow", log.single().text())
        val running = org.json.JSONObject("""{"type":"pipeline","test":{"enabled":true,"state":"running","error":null,
            "info":{"model":"sepformer-whamr","device":"cuda","gpu":"NVIDIA RTX A3000"},"monitor":"focus",
            "window_s":1.5,"hop_s":0.25,"live":{"infer_ms":41.5,"infer_p95_ms":60.0,"keeping_up":true,"hops":20,
            "skipped_hops":0,"delay_ms":750,"voice_db":[-20.5,-35.0],"pick":0,"pick_reason":"matched","error":null}}}""")
        val (live, none) = PipelineStatus.parse(running)
        assertEquals("sepformer-whamr on NVIDIA RTX A3000", live.where)
        assertEquals(41.5, live.inferMs!!, 0.0)
        assertEquals(true, live.keepingUp)
        assertEquals(listOf(-20.5, -35.0), live.voiceDb)
        assertEquals(750, live.delayMs)
        assertTrue(none.isEmpty())
    }
    @Test fun handshakeAndMediaCarryTiming() {
        val start = CaptureProtocol.start("Phone", 48000)
        assertEquals(1, start.getInt("protocol_version"))
        assertEquals("pcm_s16le", start.getString("format"))
        val message = CaptureProtocol.media("audio", "session", 9, 123456, "estimated").put("first_sample", 8640)
        assertEquals(123456, message.getLong("capture_time_us"))
        assertEquals("estimated", message.getString("timing"))
    }
}
