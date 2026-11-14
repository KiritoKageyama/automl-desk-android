package com.shadowtengu.automl.androidapp.ui.vm

import android.app.Application
import android.content.Context
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.shadowtengu.automl.androidapp.data.DesktopRemoteStatus
import com.shadowtengu.automl.androidapp.data.DesktopStatusClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class MonitorUiState {
    object Idle : MonitorUiState()
    object Connecting : MonitorUiState()
    data class Live(val status: DesktopRemoteStatus, val host: String) : MonitorUiState()
    data class Error(val message: String, val host: String) : MonitorUiState()
}

class MonitorViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val PREFS = "automl_remote_monitor"
        private const val KEY_HOST = "desktop_host"
        private const val KEY_TOKEN = "desktop_token"
        private const val DEFAULT_HOST = "10.0.2.2:8088" // emulator → host PC localhost
        private const val POLL_MS = 1500L
    }

    private val prefs = application.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val _host = mutableStateOf(prefs.getString(KEY_HOST, DEFAULT_HOST) ?: DEFAULT_HOST)
    val host: State<String> = _host

    private val _token = mutableStateOf(prefs.getString(KEY_TOKEN, "") ?: "")
    val token: State<String> = _token

    private val _uiState = mutableStateOf<MonitorUiState>(MonitorUiState.Idle)
    val uiState: State<MonitorUiState> = _uiState

    private val _polling = mutableStateOf(false)
    val polling: State<Boolean> = _polling

    private var pollJob: Job? = null

    fun setHost(value: String) {
        _host.value = value
        prefs.edit().putString(KEY_HOST, value.trim()).apply()
    }

    fun setToken(value: String) {
        _token.value = value
        prefs.edit().putString(KEY_TOKEN, value.trim()).apply()
    }

    fun startPolling() {
        if (_polling.value) return
        _polling.value = true
        _uiState.value = MonitorUiState.Connecting
        pollJob?.cancel()
        pollJob = viewModelScope.launch {
            while (isActive && _polling.value) {
                val h = _host.value
                val t = _token.value
                val result = withContext(Dispatchers.IO) {
                    DesktopStatusClient.fetchStatus(h, t)
                }
                result.onSuccess { status ->
                    _uiState.value = MonitorUiState.Live(status, DesktopStatusClient.normalizeBaseUrl(h))
                }.onFailure { e ->
                    _uiState.value = MonitorUiState.Error(
                        e.message ?: "Connection failed",
                        DesktopStatusClient.normalizeBaseUrl(h)
                    )
                }
                delay(POLL_MS)
            }
        }
    }

    fun stopPolling() {
        _polling.value = false
        pollJob?.cancel()
        pollJob = null
        if (_uiState.value !is MonitorUiState.Live) {
            _uiState.value = MonitorUiState.Idle
        }
    }

    fun refreshOnce() {
        viewModelScope.launch {
            val h = _host.value
            val t = _token.value
            _uiState.value = MonitorUiState.Connecting
            val result = withContext(Dispatchers.IO) {
                DesktopStatusClient.fetchStatus(h, t)
            }
            result.onSuccess { status ->
                _uiState.value = MonitorUiState.Live(status, DesktopStatusClient.normalizeBaseUrl(h))
            }.onFailure { e ->
                _uiState.value = MonitorUiState.Error(
                    e.message ?: "Connection failed",
                    DesktopStatusClient.normalizeBaseUrl(h)
                )
            }
        }
    }

    override fun onCleared() {
        stopPolling()
        super.onCleared()
    }
}
