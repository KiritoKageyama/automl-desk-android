package com.shadowtengu.automl.androidapp.ui.vm

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import com.example.automl_prototype_1.service.AppStateService

class SettingsViewModel : ViewModel() {

    private val appStateService = AppStateService.getInstance()

    // Scaling Strategy
    private val _scalingStrategy = mutableStateOf(appStateService.scalingStrategy ?: "Standardization")
    val scalingStrategy: State<String> = _scalingStrategy

    // Target Column
    private val _targetColumn = mutableStateOf(appStateService.targetColumnName ?: "Class")
    val targetColumn: State<String> = _targetColumn

    // K-Folds
    private val _kFolds = mutableStateOf(appStateService.kFolds)
    val kFolds: State<Int> = _kFolds

    // Tuning Enabled
    private val _isTuningEnabled = mutableStateOf(appStateService.isTuningEnabled)
    val isTuningEnabled: State<Boolean> = _isTuningEnabled

    fun setScalingStrategy(strategy: String) {
        _scalingStrategy.value = strategy
        appStateService.scalingStrategy = strategy
    }

    fun setTargetColumn(column: String) {
        _targetColumn.value = column
        appStateService.targetColumnName = column
    }

    fun setKFolds(folds: Int) {
        _kFolds.value = folds
        appStateService.kFolds = folds
    }

    fun setTuningEnabled(enabled: Boolean) {
        _isTuningEnabled.value = enabled
        appStateService.isTuningEnabled = enabled
    }
}
