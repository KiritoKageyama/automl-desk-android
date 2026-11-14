package com.shadowtengu.automl.androidapp.ui.vm

import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.automl_prototype_1.model.Dataset
import com.example.automl_prototype_1.model.ExecutionResult
import com.example.automl_prototype_1.service.AppStateService
import com.example.automl_prototype_1.service.pipeline.ExecutionService
import com.example.automl_prototype_1.service.pipeline.PreprocessorService
import com.example.automl_prototype_1.service.agentic.AgenticAdvisor
import com.example.automl_prototype_1.dataprovider.CsvDataProvider
import com.example.automl_prototype_1.service.IClassifier
import com.example.automl_prototype_1.classifier.ml.KnnClassifier
import com.example.automl_prototype_1.classifier.ml.RandomForestClassifier
import com.example.automl_prototype_1.classifier.ml.SvmClassifier
import com.example.automl_prototype_1.classifier.ml.XGBoostClassifier
import com.example.automl_prototype_1.model.PreprocessingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileReader
import java.lang.Exception
import java.lang.StringBuilder

/**
 * Represents the possible UI states for the Algorithm Selection Screen.
 */
sealed class AlgorithmSelectionState {
    object LoadingDatasetInfo : AlgorithmSelectionState()
    data class Idle(val datasetFileInfo: String) : AlgorithmSelectionState()
    object NoDatasetFile : AlgorithmSelectionState()
    object ExecutingAlgorithms : AlgorithmSelectionState()
    data class ExecutionError(val message: String) : AlgorithmSelectionState()
    object ExecutionSuccess : AlgorithmSelectionState()
}

/**
 * ViewModel for the Algorithm Selection Screen.
 */
class AlgorithmSelectionViewModel : ViewModel() {

    // --- Screen State ---
    private val _screenState = mutableStateOf<AlgorithmSelectionState>(AlgorithmSelectionState.LoadingDatasetInfo)
    val screenState: State<AlgorithmSelectionState> = _screenState

    // --- Algorithm Selection ---
    val availableAlgorithms = mutableStateMapOf(
        "Genetic Algorithm" to false,
        "IGPSO" to false,
        "WWO" to false,
        "BPSO" to false,
        "ASO" to false,
        "PLS" to false,
        "GWO" to false,
        "DE" to false,
        "Simulated Annealing" to false
    )

    // --- Classifier Selection ---
    val availableClassifiers = listOf("KNN", "Random Forest", "SVM", "XGBoost")
    private val _selectedClassifier = mutableStateOf("KNN")
    val selectedClassifier: State<String> = _selectedClassifier

    fun setClassifier(classifierName: String) {
        _selectedClassifier.value = classifierName
    }

    // --- Core Services ---
    private val appStateService = AppStateService.getInstance()
    private val executionService = ExecutionService.getInstance()
    private val preprocessorService = PreprocessorService()
    private val csvDataProvider = CsvDataProvider()

    // --- Dataset File Reference ---
    private var datasetFileReference: File? = null

    // --- Agentic AI Advisor ---
    private val agenticAdvisor = AgenticAdvisor()
    private val _adviceText = mutableStateOf("")
    val adviceText: State<String> = _adviceText

    init {
        loadDatasetFileInfoFromAppState()
    }

    private fun loadDatasetFileInfoFromAppState() {
        Log.d("AlgoSelectVM", "Checking AppStateService for dataset file reference...")
        datasetFileReference = appStateService.selectedDatasetFile

        if (datasetFileReference != null && datasetFileReference!!.exists()) {
            Log.i("AlgoSelectVM", "Dataset file found: ${datasetFileReference!!.name}. State set to Idle.")
            val info = "Using dataset: ${datasetFileReference!!.name}"
            _screenState.value = AlgorithmSelectionState.Idle(info)
        } else {
            Log.e("AlgoSelectVM", "Dataset file reference invalid or file missing. State set to NoDatasetFile.")
            datasetFileReference = null
            appStateService.selectedDatasetFile = null
            _screenState.value = AlgorithmSelectionState.NoDatasetFile
        }
    }

    fun toggleAlgorithmSelection(algorithmName: String) {
        if (availableAlgorithms.containsKey(algorithmName)) {
            val currentState = availableAlgorithms[algorithmName] ?: false
            availableAlgorithms[algorithmName] = !currentState
            Log.d("AlgoSelectVM", "Toggled selection for $algorithmName to ${!currentState}")
        }
    }

    fun getSelectedAlgorithms(): List<String> {
        return availableAlgorithms.filter { it.value }.keys.toList()
    }

    fun getAdvice() {
        if (datasetFileReference == null || !datasetFileReference!!.exists()) {
            _adviceText.value = "Please select a dataset first."
            return
        }

        viewModelScope.launch {
            try {
                val advice = withContext(Dispatchers.IO) {
                    val dataset = FileReader(datasetFileReference!!).use { reader ->
                        csvDataProvider.loadDataset(reader)
                    }
                    val suggestions = agenticAdvisor.suggestPipelines(dataset, datasetFileReference!!.name)
                    
                    val sb = StringBuilder()
                    sb.append("--- Agentic AI Recommendations ---\n\n")
                    
                    if (suggestions.isEmpty()) {
                        sb.append("No specific recommendations found. Try standard algorithms like Random Forest.")
                    } else {
                        suggestions.forEachIndexed { index, s ->
                            sb.append("${index + 1}. ${s.algorithmName} + ${s.classifierName} (Confidence: ${(s.confidence * 100).toInt()}%)\n")
                            sb.append("   Reason: ${s.reasoning}\n\n")
                        }
                    }
                    sb.toString()
                }
                _adviceText.value = advice
            } catch (e: Exception) {
                _adviceText.value = "Error generating advice: ${e.localizedMessage}"
                Log.e("AlgoSelectVM", "Error generating advice", e)
            }
        }
    }

    fun runSelectedAlgorithms() {
        val selectedAlgos = getSelectedAlgorithms()
        if (selectedAlgos.isEmpty()) {
            Log.w("AlgoSelectVM", "Execute requested but no algorithms selected.")
            _screenState.value = AlgorithmSelectionState.ExecutionError("Please select at least one algorithm.")
            viewModelScope.launch { kotlinx.coroutines.delay(2000); loadDatasetFileInfoFromAppState() }
            return
        }

        if (datasetFileReference == null || !datasetFileReference!!.exists()) {
            Log.e("AlgoSelectVM", "Execute requested but dataset file reference is invalid or file missing!")
            _screenState.value = AlgorithmSelectionState.NoDatasetFile
            return
        }

        Log.i("AlgoSelectVM", "Executing algorithms: $selectedAlgos on file ${datasetFileReference!!.name}")
        _screenState.value = AlgorithmSelectionState.ExecutingAlgorithms

        viewModelScope.launch {
            var loadedDataset: Dataset? = null
            var finalDataset: Dataset? = null
            var executionResults: List<ExecutionResult>? = null

            try {
                loadedDataset = withContext(Dispatchers.IO) {
                    Log.d("AlgoSelectVM", "Loading Dataset object via CsvDataProvider...")
                    FileReader(datasetFileReference!!).use { reader ->
                        csvDataProvider.loadDataset(reader)
                    }
                }
                Log.i("AlgoSelectVM", "Dataset object loaded successfully.")
            } catch (loadEx: Exception) {
                Log.e("AlgoSelectVM", "Failed to load Dataset object from file", loadEx)
                _screenState.value = AlgorithmSelectionState.ExecutionError("Failed to load data: ${loadEx.localizedMessage}")
                return@launch
            }

            if (loadedDataset != null) {
                try {
                    finalDataset = withContext(Dispatchers.Default) {
                        Log.d("AlgoSelectVM", "Running Preprocessing Pipeline...")
                        val target = appStateService.targetColumnName ?: "Class"
                        val config = appStateService.preprocessingConfig ?: PreprocessingConfig().also {
                            it.scalingStrategy = appStateService.scalingStrategy ?: "Standardization"
                        }
                        preprocessorService.runPreprocessingPipeline(loadedDataset!!, target, config)
                    }
                    Log.i("AlgoSelectVM", "Preprocessing complete.")
                } catch (prepEx: Exception) {
                    Log.e("AlgoSelectVM", "Failed to preprocess data", prepEx)
                    _screenState.value = AlgorithmSelectionState.ExecutionError("Preprocessing failed: ${prepEx.localizedMessage}")
                    return@launch
                }
            }

            if (finalDataset != null) {
                try {
                    executionResults = withContext(Dispatchers.Default) {
                        Log.d("AlgoSelectVM", "Calling executionService.executeAlgorithms...")
                        
                        val classifier: IClassifier = when (_selectedClassifier.value) {
                            "Random Forest" -> RandomForestClassifier()
                            "SVM" -> SvmClassifier()
                            "XGBoost" -> XGBoostClassifier()
                            else -> KnnClassifier()
                        }
                        val config: Map<String, Any> = mapOf("classifier" to classifier)

                        executionService.executeAlgorithms(finalDataset!!, selectedAlgos, config)
                    }
                    Log.i("AlgoSelectVM", "Execution completed. Results count: ${executionResults?.size ?: "null"}")

                    appStateService.executionResults = executionResults
                    _screenState.value = AlgorithmSelectionState.ExecutionSuccess

                } catch (execEx: Exception) {
                    Log.e("AlgoSelectVM", "Exception during algorithm execution", execEx)
                    _screenState.value = AlgorithmSelectionState.ExecutionError("Execution failed: ${execEx.localizedMessage}")
                    appStateService.executionResults = null
                }
            } else {
                Log.e("AlgoSelectVM", "Dataset object was null after loading attempt, cannot execute.")
                _screenState.value = AlgorithmSelectionState.ExecutionError("Internal error: Failed to prepare data.")
                appStateService.executionResults = null
            }
        }
    }

    fun returnToIdle() {
        loadDatasetFileInfoFromAppState()
    }
}