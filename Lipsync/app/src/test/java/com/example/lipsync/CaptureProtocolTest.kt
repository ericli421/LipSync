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
    @Test fun handshakeAndMediaCarryTiming() {
        val start = CaptureProtocol.start("Phone", 48000)
        assertEquals(1, start.getInt("protocol_version"))
        assertEquals("pcm_s16le", start.getString("format"))
        val message = CaptureProtocol.media("audio", "session", 9, 123456, "estimated").put("first_sample", 8640)
        assertEquals(123456, message.getLong("capture_time_us"))
        assertEquals("estimated", message.getString("timing"))
    }
}
