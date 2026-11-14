package com.shadowtengu.automl.androidapp.data

import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

/**
 * Desktop remote status payload from GET /api/status on RemoteStatusServer (:8088).
 */
data class DesktopRemoteStatus(
    val timestamp: Long = 0L,
    val datasetName: String? = null,
    val domain: String? = null,
    val runningAlgo: String? = null,
    val phase: String = "IDLE",
    val progressPct: Double = 0.0,
    val accuracySoFar: Double? = null,
    val globalBestAlgo: String? = null,
    val globalBestScore: Double? = null,
    val queue: List<String> = emptyList(),
    val useGpu: Boolean = false,
    val gpuName: String? = null,
    val gpuServerUp: Boolean = false,
    val watchingJob: String? = null,
    val rawJson: String = ""
)

object DesktopStatusClient {

    private const val TAG = "DesktopStatusClient"
    private const val CONNECT_TIMEOUT_MS = 2500
    private const val READ_TIMEOUT_MS = 4000

    fun normalizeBaseUrl(hostOrUrl: String): String {
        var s = hostOrUrl.trim()
        if (s.isEmpty()) return "http://10.0.2.2:8088"
        if (!s.startsWith("http://") && !s.startsWith("https://")) {
            s = "http://$s"
        }
        // Strip trailing slash
        while (s.endsWith("/")) s = s.dropLast(1)
        // If user only gave host without port, assume 8088
        val afterProto = s.removePrefix("http://").removePrefix("https://")
        if (!afterProto.contains(":")) {
            s = "$s:8088"
        }
        return s
    }

    fun fetchStatus(hostOrUrl: String, token: String? = null): Result<DesktopRemoteStatus> {
        return try {
            val base = normalizeBaseUrl(hostOrUrl)
            val url = URL("$base/api/status")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                setRequestProperty("Accept", "application/json")
                val t = token?.trim().orEmpty()
                if (t.isNotEmpty()) {
                    setRequestProperty("X-AutoML-Token", t)
                }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = BufferedReader(InputStreamReader(stream)).use { it.readText() }
            if (code !in 200..299) {
                return Result.failure(IllegalStateException("HTTP $code: $body"))
            }
            Result.success(parse(body))
        } catch (e: Exception) {
            Log.w(TAG, "fetchStatus failed: ${e.message}")
            Result.failure(e)
        }
    }

    fun ping(hostOrUrl: String): Result<Boolean> {
        return try {
            val base = normalizeBaseUrl(hostOrUrl)
            val url = URL("$base/api/health")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
            }
            val ok = conn.responseCode in 200..299
            Result.success(ok)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun parse(json: String): DesktopRemoteStatus {
        val o = JSONObject(json)
        val queue = mutableListOf<String>()
        val qArr = o.optJSONArray("queue")
        if (qArr != null) {
            for (i in 0 until qArr.length()) {
                queue.add(qArr.optString(i))
            }
        }
        return DesktopRemoteStatus(
            timestamp = o.optLong("timestamp", 0L),
            datasetName = o.optStringOrNull("datasetName"),
            domain = o.optStringOrNull("domain"),
            runningAlgo = o.optStringOrNull("runningAlgo"),
            phase = o.optString("phase", "IDLE"),
            progressPct = o.optDouble("progressPct", 0.0),
            accuracySoFar = o.optDoubleOrNull("accuracySoFar"),
            globalBestAlgo = o.optStringOrNull("globalBestAlgo"),
            globalBestScore = o.optDoubleOrNull("globalBestScore"),
            queue = queue,
            useGpu = o.optBoolean("useGpu", false),
            gpuName = o.optStringOrNull("gpuName"),
            gpuServerUp = o.optBoolean("gpuServerUp", false),
            watchingJob = o.optStringOrNull("watchingJob"),
            rawJson = json
        )
    }

    private fun JSONObject.optStringOrNull(key: String): String? {
        if (!has(key) || isNull(key)) return null
        val v = optString(key, "")
        return v.ifBlank { null }
    }

    private fun JSONObject.optDoubleOrNull(key: String): Double? {
        if (!has(key) || isNull(key)) return null
        return try {
            getDouble(key)
        } catch (_: Exception) {
            null
        }
    }
}
