package com.shadowtengu.automl.androidapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.shadowtengu.automl.androidapp.ui.vm.SettingsViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    navController: NavController,
    viewModel: SettingsViewModel = viewModel()
) {
    val scalingStrategy by viewModel.scalingStrategy
    val targetColumn by viewModel.targetColumn
    val kFolds by viewModel.kFolds
    val isTuningEnabled by viewModel.isTuningEnabled

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            // Scaling Strategy Section
            Text("Scaling Strategy", style = MaterialTheme.typography.titleMedium)
            Column {
                listOf("Standardization", "Normalization").forEach { text ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = (text == scalingStrategy),
                                onClick = { viewModel.setScalingStrategy(text) },
                                role = Role.RadioButton
                            )
                            .padding(vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = (text == scalingStrategy),
                            onClick = null // null recommended for accessibility with selectable
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(text = text)
                    }
                }
            }

            Divider()

            // Target Column Section
            Text("Target Column", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = targetColumn,
                onValueChange = { viewModel.setTargetColumn(it) },
                label = { Text("Target Column Name") },
                modifier = Modifier.fillMaxWidth()
            )

            Divider()

            // K-Folds Section
            Text("K-Folds Cross Validation: $kFolds", style = MaterialTheme.typography.titleMedium)
            Slider(
                value = kFolds.toFloat(),
                onValueChange = { viewModel.setKFolds(it.toInt()) },
                valueRange = 2f..10f,
                steps = 7 // (10-2)-1 = 7 steps between 2 and 10
            )

            Divider()

            // Tuning Section
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Enable Hyperparameter Tuning", style = MaterialTheme.typography.titleMedium)
                Switch(
                    checked = isTuningEnabled,
                    onCheckedChange = { viewModel.setTuningEnabled(it) }
                )
            }
        }
    }
}
