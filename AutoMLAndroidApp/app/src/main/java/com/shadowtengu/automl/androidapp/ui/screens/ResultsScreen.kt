package com.shadowtengu.automl.androidapp.ui.screens

import android.util.Log
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.automl_prototype_1.model.ExecutionResult
import com.shadowtengu.automl.androidapp.ui.theme.*
import com.shadowtengu.automl.androidapp.ui.vm.ResultsViewModel
import com.shadowtengu.automl.androidapp.ui.vm.ResultsScreenState
import java.util.Locale

/**
 * Composable function for the Results Screen.
 * Displays the results of the executed algorithms in a modern, card-based layout.
 */
@Composable
fun ResultsScreen(
    navController: NavController,
    viewModel: ResultsViewModel = viewModel()
) {
    Log.d("WHICH_SCREEN", "--- Rendering ResultsScreen ---")
    val screenState by viewModel.screenState

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header with gradient background
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(
                        Brush.horizontalGradient(
                            colors = listOf(GradientStart, GradientEnd)
                        )
                    )
                    .padding(20.dp)
            ) {
                Column {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(32.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "Execution Results",
                        style = MaterialTheme.typography.headlineMedium,
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        "Algorithm Performance Analysis",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White.copy(alpha = 0.9f)
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            when (val state = screenState) {
                is ResultsScreenState.Loading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                "Loading results...",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                        }
                    }
                }
                is ResultsScreenState.NoResults -> {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer
                        ),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                imageVector = Icons.Default.Warning,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(48.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                "No results found",
                                style = MaterialTheme.typography.titleLarge,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = { navController.popBackStack() },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.error
                                )
                            ) {
                                Icon(Icons.Default.ArrowBack, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Go Back")
                            }
                        }
                    }
                }
                is ResultsScreenState.Error -> {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer
                        ),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(48.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                "Error loading results",
                                style = MaterialTheme.typography.titleLarge,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Text(
                                state.message,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = { viewModel.refreshResults() },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.error
                                )
                            ) {
                                Icon(Icons.Default.Refresh, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Retry")
                            }
                        }
                    }
                }
                is ResultsScreenState.Success -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(state.results) { result ->
                            ModernResultCard(result = result)
                        }
                    }
                }
            }
        }
    }
}

/**
 * Modern card-based composable for displaying a single algorithm result.
 */
@Composable
private fun ModernResultCard(result: ExecutionResult) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(
                animationSpec = spring(
                    dampingRatio = Spring.DampingRatioMediumBouncy,
                    stiffness = Spring.StiffnessLow
                )
            ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            // Algorithm Name Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = result.algorithmName,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                // Accuracy Badge
                AccuracyBadge(result.accuracy)
            }

            Spacer(modifier = Modifier.height(16.dp))
            Divider(color = MaterialTheme.colorScheme.outlineVariant)
            Spacer(modifier = Modifier.height(16.dp))

            // Metrics Grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricItem(
                    icon = Icons.Default.Star,
                    label = "Accuracy",
                    value = formatMetric(result.accuracy),
                    color = getAccuracyColor(result.accuracy),
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    icon = Icons.Default.DateRange,
                    label = "AUC-ROC",
                    value = formatMetric(result.aucRoc),
                    color = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricItem(
                    icon = Icons.Default.Info,
                    label = "Loss",
                    value = formatMetric(result.loss),
                    color = MaterialTheme.colorScheme.tertiary,
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    icon = Icons.Default.Done,
                    label = "Time (ms)",
                    value = result.executionTimeMs.toString(),
                    color = Color(0xFF8B5CF6),
                    modifier = Modifier.weight(1f)
                )
            }

            // Additional metrics if available
            if (result.selectedFeatures > 0) {
                Spacer(modifier = Modifier.height(12.dp))
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        SmallMetricChip(
                            label = "Features",
                            value = "${result.selectedFeatures}/${result.originalFeatures}"
                        )
                        if (!result.precision.isNaN() && result.precision > 0.0) {
                            SmallMetricChip(
                                label = "Precision",
                                value = formatMetric(result.precision)
                            )
                        }
                    }
                    if (!result.recall.isNaN() && result.recall > 0.0) {
                        SmallMetricChip(
                            label = "Recall",
                            value = formatMetric(result.recall)
                        )
                    }
                }
            }
            
            // Fitness History Chart
            if (result.fitnessHistory != null && result.fitnessHistory.isNotEmpty()) {
                Spacer(modifier = Modifier.height(16.dp))
                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = "Fitness History",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(12.dp))
                FitnessHistoryChart(fitnessHistory = result.fitnessHistory)
            }
            
            // Confusion Matrix
            if (result.confusionMatrix != null && result.confusionMatrix.isNotEmpty()) {
                Spacer(modifier = Modifier.height(16.dp))
                Divider(color = MaterialTheme.colorScheme.outlineVariant)
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = "Confusion Matrix",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(12.dp))
                ConfusionMatrixView(confusionMatrix = result.confusionMatrix)
            }
        }
    }
}

/**
 * Helper function to format metrics, handling NaN values properly.
 */
private fun formatMetric(value: Double): String {
    return if (value.isNaN() || value.isInfinite()) {
        "N/A"
    } else {
        String.format(Locale.US, "%.4f", value)
    }
}

/**
 * Get color based on accuracy value.
 */
@Composable
private fun getAccuracyColor(accuracy: Double): Color {
    return when {
        accuracy.isNaN() || accuracy.isInfinite() -> MaterialTheme.colorScheme.onSurfaceVariant
        accuracy >= 0.9 -> Success
        accuracy >= 0.7 -> Warning
        else -> Error
    }
}

/**
 * Accuracy badge component.
 */
@Composable
private fun AccuracyBadge(accuracy: Double) {
    val displayText = if (accuracy.isNaN() || accuracy.isInfinite()) {
        "N/A"
    } else {
        "${(accuracy * 100).toInt()}%"
    }
    
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = getAccuracyColor(accuracy).copy(alpha = 0.15f)
    ) {
        Text(
            text = displayText,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelLarge,
            color = getAccuracyColor(accuracy),
            fontWeight = FontWeight.Bold
        )
    }
}

/**
 * Individual metric item with icon.
 */
@Composable
private fun MetricItem(
    icon: ImageVector,
    label: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = color.copy(alpha = 0.1f)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Small metric chip for additional metrics.
 */
@Composable
private fun SmallMetricChip(label: String, value: String) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.secondaryContainer
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "$label: ",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                maxLines = 1,
                softWrap = false
            )
            Text(
                text = value,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                maxLines = 1,
                softWrap = false
            )
        }
    }
}

/**
 * Fitness History Line Chart using MPAndroidChart.
 */
@Composable
private fun FitnessHistoryChart(fitnessHistory: List<Double>) {
    val primary = MaterialTheme.colorScheme.primary
    val onSurface = MaterialTheme.colorScheme.onSurface
    val surfaceVariant = MaterialTheme.colorScheme.surfaceVariant
    
    AndroidView(
        modifier = Modifier
            .fillMaxWidth()
            .height(250.dp),
        factory = { context ->
            com.github.mikephil.charting.charts.LineChart(context).apply {
                description.isEnabled = false
                setTouchEnabled(true)
                isDragEnabled = true
                setScaleEnabled(true)
                setPinchZoom(true)
                setDrawGridBackground(false)
                legend.isEnabled = false
                
                // Configure X-axis
                xAxis.apply {
                    position = com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM
                    setDrawGridLines(false)
                    textColor = onSurface.hashCode()
                    granularity = 1f
                }
                
                // Configure left Y-axis
                axisLeft.apply {
                    setDrawGridLines(true)
                    gridColor = surfaceVariant.hashCode()
                    textColor = onSurface.hashCode()
                }
                
                // Disable right Y-axis
                axisRight.isEnabled = false
                
                // Prepare data
                val entries = fitnessHistory.mapIndexed { index, value ->
                    com.github.mikephil.charting.data.Entry(index.toFloat(), value.toFloat())
                }
                
                val dataSet = com.github.mikephil.charting.data.LineDataSet(entries, "Fitness").apply {
                    color = primary.hashCode()
                    lineWidth = 3f
                    setDrawCircles(false)
                    setDrawValues(false)
                    mode = com.github.mikephil.charting.data.LineDataSet.Mode.CUBIC_BEZIER
                    setDrawFilled(true)
                    fillColor = primary.hashCode()
                    fillAlpha = 50
                }
                
                data = com.github.mikephil.charting.data.LineData(dataSet)
                animateX(800)
                invalidate()
            }
        }
    )
}

/**
 * Confusion Matrix Heatmap using Compose Canvas.
 */
@Composable
private fun ConfusionMatrixView(confusionMatrix: Array<IntArray>) {
    val primary = MaterialTheme.colorScheme.primary
    val onSurface = MaterialTheme.colorScheme.onSurface
    val surface = MaterialTheme.colorScheme.surface
    
    val size = confusionMatrix.size
   if (size == 0) return
    
    // Calculate max value for color scaling
    val maxValue = confusionMatrix.flatMap { it.toList() }.maxOrNull() ?: 1
    
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = surface,
        tonalElevation = 2.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Labels
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "Predicted →",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Rotated "Actual" label
                Text(
                    text = "↓ Actual",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(end = 8.dp)
                )
                
                // Matrix grid
                Column(
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    confusionMatrix.forEachIndexed { rowIndex, row ->
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            row.forEachIndexed { colIndex, value ->
                                val intensity = if (maxValue > 0) value.toFloat() / maxValue else 0f
                                val isCorrect = rowIndex == colIndex
                                
                                Surface(
                                    modifier = Modifier
                                        .size(if (size <= 2) 80.dp else 60.dp),
                                    shape = RoundedCornerShape(8.dp),
                                    color = if (isCorrect) {
                                        Color(0xFF10B981).copy(alpha = 0.2f + (intensity * 0.6f))
                                    } else {
                                        Color(0xFFEF4444).copy(alpha = 0.1f + (intensity * 0.4f))
                                    }
                                ) {
                                    Box(
                                        contentAlignment = Alignment.Center,
                                        modifier = Modifier.fillMaxSize()
                                    ) {
                                        Text(
                                            text = value.toString(),
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = if (intensity > 0.5f) Color.White else onSurface
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Legend
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    modifier = Modifier.size(16.dp),
                    shape = RoundedCornerShape(4.dp),
                    color = Color(0xFF10B981).copy(alpha = 0.6f)
                ) {}
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Correct",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.width(16.dp))
                Surface(
                    modifier = Modifier.size(16.dp),
                    shape = RoundedCornerShape(4.dp),
                    color = Color(0xFFEF4444).copy(alpha = 0.4f)
                ) {}
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Incorrect",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}