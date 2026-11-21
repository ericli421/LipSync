package com.example.lipsync

import android.annotation.SuppressLint
import android.content.Context
import android.media.*
import android.os.SystemClock
import java.util.concurrent.ArrayBlockingQueue
import kotlin.concurrent.thread
import kotlin.math.sqrt

class PhoneAudio(private val context: Context) {
    private var recorder: AudioRecord? = null
    private var reader: Thread? = null
    private var writer: Thread? = null
    private var track: AudioTrack? = null
    private val playback = ArrayBlockingQueue<ByteArray>(10)
    @Volatile private var recording = false
    @Volatile var monitoring = false
    private var rate = 48000
    private var resetPlayback = false
    private var playbackGeneration = 0L
    private val playbackLock = Any()

    @SuppressLint("MissingPermission")
    fun prepare(): Int {
        val manager = context.getSystemService(AudioManager::class.java)
        val source = if (manager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true")
            MediaRecorder.AudioSource.UNPROCESSED else MediaRecorder.AudioSource.VOICE_RECOGNITION
        for (candidate in listOf(48000, 16000, 44100)) {
            val minimum = AudioRecord.getMinBufferSize(candidate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minimum <= 0) continue
            val record = try { AudioRecord.Builder().setAudioSource(source).setAudioFormat(AudioFormat.Builder()
                .setSampleRate(candidate).setChannelMask(AudioFormat.CHANNEL_IN_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(maxOf(minimum, candidate / 5 * 2)).build() } catch (_: Exception) { continue }
            if (record.state == AudioRecord.STATE_INITIALIZED) {
                recorder = record; rate = candidate
                manager.getDevices(AudioManager.GET_DEVICES_INPUTS).firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }?.let { record.preferredDevice = it }
                return rate
            }
            record.release()
        }
        error("No supported mono PCM microphone input")
    }

    @SuppressLint("MissingPermission")
    fun start(send: (ByteArray, Long, Long, Long, String, Float, String) -> Unit, error: (String) -> Unit) {
        val record = checkNotNull(recorder)
        record.startRecording()
        check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "Microphone did not start" }
        recording = true
        reader = thread(name = "phone-microphone") {
            val clock = AudioSampleClock(rate)
            val timestamp = AudioTimestamp()
            val estimatedAnchor = SystemClock.elapsedRealtimeNanos()
            try {
                val block = ByteArray(clock.blockSamples * 2)
                while (recording) {
                    var offset = 0
                    while (recording && offset < block.size) {
                        val count = record.read(block, offset, block.size - offset, AudioRecord.READ_BLOCKING)
                        check(count > 0) { "Microphone read failed ($count)" }
                        offset += count
                    }
                    if (!recording) break
                    val (sequence, firstSample) = clock.advance()
                    val valid = record.getTimestamp(timestamp, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS
                    val captureUs = if (valid) clock.timeUs(firstSample, timestamp.framePosition, timestamp.nanoTime)
                        else clock.timeUs(firstSample, 0, estimatedAnchor)
                    var sum = 0.0
                    for (i in block.indices step 2) {
                        val sample = ((block[i].toInt() and 255) or (block[i + 1].toInt() shl 8)).toShort() / 32768.0
                        sum += sample * sample
                    }
                    val input = record.routedDevice?.let { "${it.productName} (input ${it.type})" } ?: "Input route pending"
                    send(block.copyOf(), sequence, firstSample, captureUs.coerceAtLeast(0), if (valid) "device" else "estimated", sqrt(sum / clock.blockSamples).toFloat(), input)
                }
            } catch (exception: Exception) { if (recording) error(exception.message ?: "Microphone failed") }
        }
        writer = thread(name = "phone-headphones") {
            try {
                val minimum = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
                val output = AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()).setAudioFormat(AudioFormat.Builder()
                    .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                    .setTransferMode(AudioTrack.MODE_STREAM).setBufferSizeInBytes(maxOf(minimum, rate / 10 * 2)).build()
                synchronized(playbackLock) { track = output }
                output.play()
                while (recording) {
                    val epoch = synchronized(playbackLock) { playbackGeneration }
                    val bytes = playback.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS)
                    synchronized(playbackLock) {
                        if (resetPlayback) { output.pause(); output.flush(); output.play(); resetPlayback = false }
                        if (bytes != null && monitoring && epoch == playbackGeneration) {
                            val written = output.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
                            if (written != bytes.size) { playback.clear(); resetPlayback = true }
                        }
                    }
                }
            } catch (exception: Exception) { if (recording) error(exception.message ?: "Audio playback failed") }
            finally { synchronized(playbackLock) { track?.release(); track = null } }
        }
    }

    fun monitor(enabled: Boolean) { synchronized(playbackLock) { monitoring = enabled; playback.clear(); resetPlayback = true; playbackGeneration++ } }
    fun enqueue(bytes: ByteArray) { synchronized(playbackLock) {
        if (!monitoring) return
        if (!playback.offer(bytes)) { playback.clear(); playback.offer(bytes); resetPlayback = true; playbackGeneration++ }
    } }
    fun discontinuity() { synchronized(playbackLock) { playback.clear(); resetPlayback = true; playbackGeneration++ } }
    fun stop() {
        recording = false; monitoring = false; playback.clear()
        try { recorder?.stop() } catch (_: IllegalStateException) { }
        reader?.join(500); writer?.join(500)
        recorder?.release(); recorder = null
    }
}
