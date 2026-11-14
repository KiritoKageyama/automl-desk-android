package com.shadowtengu.automl.androidapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.shadowtengu.automl.androidapp.ui.theme.GradientEnd
import com.shadowtengu.automl.androidapp.ui.theme.GradientStart
import com.shadowtengu.automl.androidapp.ui.vm.MonitorUiState
import com.shadowtengu.automl.androidapp.ui.vm.MonitorViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MonitorScreen(
    navController: NavController,
    viewModel: MonitorViewModel = viewModel()
) {
    val host by viewModel.host
    val token by viewModel.token
    val uiState by viewModel.uiState
    val polling by viewModel.polling

    DisposableEffect(Unit) {
        viewModel.startPolling()
        onDispose { viewModel.stopPolling() }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Desktop Monitor") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refreshOnce() }) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Brush.horizontalGradient(listOf(GradientStart, GradientEnd)))
                    .padding(20.dp)
            ) {
                Column {
                    Text(
                        "Remote AutoML Status",
                        style = MaterialTheme.typography.headlineSmall,
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        "Polls desktop :8088 /api/status over LAN",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White.copy(alpha = 0.9f)
                    )
                }
            }

            OutlinedTextField(
                value = host,
                onValueChange = { viewModel.setHost(it) },
                label = { Text("Desktop host (IP or URL)") },
                supportingText = {
                    Text("Emulator: 10.0.2.2:8088 · Phone: your-PC-LAN-IP:8088")
                },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            OutlinedTextField(
                value = token,
                onValueChange = { viewModel.setToken(it) },
                label = { Text("Desktop token (X-AutoML-Token)") },
                supportingText = {
                    Text("Copy from desktop Help, or ~/.automl_remote_token")
                },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = { viewModel.startPolling() },
                    enabled = !polling,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Start")
                }
                OutlinedButton(
                    onClick = { viewModel.stopPolling() },
                    enabled = polling,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.Close, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Stop")
                }
            }

            when (val state = uiState) {
                is MonitorUiState.Idle -> {
                    Text("Enter desktop IP and press Start.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                is MonitorUiState.Connecting -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                        Spacer(Modifier.width(12.dp))
                        Text("Connecting…")
                    }
                }
                is MonitorUiState.Error -> {
                    StatusCard(title = "Connection", accent = MaterialTheme.colorScheme.error) {
                        Text(state.message, color = MaterialTheme.colorScheme.error)
                        Spacer(Modifier.height(8.dp))
                        Text("Tried: ${state.host}", style = MaterialTheme.typography.bodySmall)
                        Text(
                            "Ensure desktop app is running and Windows Firewall allows TCP 8088.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
                is MonitorUiState.Live -> {
                    val s = state.status
                    StatusCard(title = "Link", accent = Color(0xFF2E7D32)) {
                        Text("Connected · ${state.host}", fontWeight = FontWeight.Medium)
                        Text("Phase: ${s.phase}", style = MaterialTheme.typography.bodyMedium)
                    }

                    StatusCard(title = "Job", accent = MaterialTheme.colorScheme.primary) {
                        Text("Dataset: ${s.datasetName ?: "—"}")
                        Text("Domain: ${s.domain ?: "—"}")
                        Text("Running: ${s.runningAlgo ?: "Idle"}", fontWeight = FontWeight.SemiBold)
                        if (s.watchingJob != null) {
                            Text("Watchdog: ${s.watchingJob}")
                        }
                        Spacer(Modifier.height(12.dp))
                        LinearProgressIndicator(
                            progress = (s.progressPct / 100.0).toFloat().coerceIn(0f, 1f),
                            modifier = Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(4.dp))
                        )
                        Text("${"%.1f".format(s.progressPct)}%", style = MaterialTheme.typography.labelLarge)
                    }

                    StatusCard(title = "Scores", accent = MaterialTheme.colorScheme.tertiary) {
                        Text(
                            "Accuracy so far: ${s.accuracySoFar?.let { "%.4f".format(it) } ?: "—"}",
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            "Global best: ${s.globalBestAlgo ?: "—"} " +
                                (s.globalBestScore?.let { "(${"%.4f".format(it)})" } ?: "")
                        )
                        if (s.queue.isNotEmpty()) {
                            Spacer(Modifier.height(8.dp))
                            Text("Queue: ${s.queue.joinToString(", ")}")
                        }
                    }

                    StatusCard(title = "GPU", accent = if (s.gpuServerUp) Color(0xFF1565C0) else Color.Gray) {
                        Text("GPU enabled: ${s.useGpu}")
                        Text("Device: ${s.gpuName ?: "—"}")
                        Text("Python GPU server: ${if (s.gpuServerUp) "UP" else "DOWN"}")
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusCard(
    title: String,
    accent: Color,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, color = accent, fontWeight = FontWeight.Bold)
            content()
        }
    }
}
