# Codebase Audit Record
Root: C:/Users/arinb/IdeaProjects/New_Automl/Automl_desk+Andriod/automl_prototype_1

## Module: automl_core

### Package: com.example.automl_prototype_1.algorithm
#### AblationAwareGeneticAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 1 classes (e.g., ExecutionService)
- **Dependencies**: 5 imports
- **Logic/Math**: Standard Algorithm Implementation

#### AbstractFeatureSelectionAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 20 imports
- **Logic/Math**: Possible Math Logic identified: Pre-computed feature metadata to avoid repeated scanning and potential; Pre-computed, SCALED data for each fold (CPU Cache); Pre-compute low cardinality features globally to avoid leakage and improve...

#### AtomSearchOptimization
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, DashboardController)
- **Dependencies**: 3 imports
- **Logic/Math**: Physics-based: 1) Interaction Force: F[i] = sum(rand * beta * (x[k] - x[i])). 2) Constraint Force: G[i] = alpha * (x[best] - x[i]). 3) Acceleration: a[i] = F[i]/m[i]. 4) Motion: x[t+1] = x[t] + v[t+1]. Uses L-J potential model mechanics.

#### BinaryParticleSwarmOptimization
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 7 imports
- **Logic/Math**: Standard PSO: 1) v[t+1] = w*v[t] + c1*r1*(pbest-x) + c2*r2*(gbest-x). 2) Position Update: x[t+1] = 1 if rand < Sigmoid(v[t+1]) else 0. Uses generic sigmoid transfer function.

#### ChromosomeKey
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: Standard Algorithm Implementation

#### DifferentialEvolution
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 1 classes (e.g., ExecutionService)
- **Dependencies**: 7 imports
- **Logic/Math**: Evolutionary: 1) Mutation (DE/rand/1): v = x[r1] + F*(x[r2]-x[r3]). 2) Crossover: Binomial crossover with rate CR. 3) Selection: Greedy (child replaces parent if fitness is better). 4) Discretization: Sigmoid mapping for binary selection.

#### GeneticAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 4 imports
- **Logic/Math**: Standard GA: 1) Selection: Tournament Selection. 2) Crossover: Single-point or Uniform. 3) Mutation: Bit-flip mutation with low probability (1/N).

#### GreyWolfOptimizer
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 1 classes (e.g., ExecutionService)
- **Dependencies**: 6 imports
- **Logic/Math**: Swarm Intelligence: 1) Hierarchy: Alpha, Beta, Delta lead pack. 2) Encircling: D = |C*X_p - X|. 3) Hunting: X[new] = (X1 + X2 + X3)/3 where X_i is position vector derived from Alpha/Beta/Delta relative vectors. 4) Decay: 'a' parameter decreases linearly 2->0.

#### IGPSO
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 5 imports
- **Logic/Math**: Hybrid Optimization: 1) Chaotic Init (Logistic Map): x[k+1] = 4x[k](1-x[k]). 2) Velocity: v[t+1] = w*v[t] + c1*r1*(pbest-x) + c2*r2*(gbest-x). 3) Importance-Guided: v'[t+1] = I[j] * v[t+1] where I[j] is variance-based feature importance. 4) Update: Sigmoid transfer function S(v) = 1/(1+exp(-I[j]*v)).

#### ManualSelectionAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 9 imports
- **Logic/Math**: No Optimization: User explicitly selects feature subset indices. Bypasses all heuristic search steps.

#### NoOptimizationAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 1 classes (e.g., ExecutionService)
- **Dependencies**: 5 imports
- **Logic/Math**: Baseline: Selects ALL features or NO features depending on configuration. Used for control in experiments.

#### OptimizationAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 3 classes (e.g., AlgorithmRecommendation, ExecutionService, RecommendationService)
- **Dependencies**: 4 imports
- **Logic/Math**: Possible Math Logic identified: Interface for all optimization algorithms...

#### ParallelLocalSearchAlgorithm
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 5 imports
- **Logic/Math**: Stochastic: Independent threads explore neighborhood by flipping random bits. Best found solution is synchronized globally. Simple Hill-Climbing parallelized.

#### SimulatedAnnealing
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 1 classes (e.g., ExecutionService)
- **Dependencies**: 5 imports
- **Logic/Math**: Thermodynamic: 1) Acceptance Probability: P = exp(-(E_new - E_old) / T). 2) Cooling Schedule: T[k+1] = alpha * T[k]. Accepts worse solutions early to escape local optima.

#### WaterWaveOptimization
- **Status**: Implemented
- **High Level Feature**: AutoML Core Engine
- **Specific Contribution**: Machine Learning Algorithm
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 6 imports
- **Logic/Math**: Wave Mechanics: 1) Propagation: x' = x + lambda * rand(-1,1). 2) Refraction: lambda' = lambda * (h'/h) where h is wave height (fitness). 3) Breaking: If h > h_max, wave 'breaks' into sub-waves (managed as re-initialization).


### Package: com.example.automl_prototype_1.classifier
#### BertClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### CnnClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### CustomScriptClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### GenericHuggingFaceClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., ModelHubViewController)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### LstmClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### MlpClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.classifier.cpu
#### KnnClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 4 classes (e.g., ClassifierRegistry, BehavioralCloningService, ExecutionService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### LogisticRegression
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### RandomForestClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 3 classes (e.g., AbstractFeatureSelectionAlgorithm, ClassifierRegistry, RecommendationService)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### SvmClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 3 classes (e.g., AbstractFeatureSelectionAlgorithm, ClassifierRegistry, RecommendationService)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### XGBoostClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., AbstractFeatureSelectionAlgorithm, ClassifierRegistry)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.classifier.gpu
#### GpuKnnClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### GpuLogisticRegression
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### GpuPythonCatBoostClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### GpuPythonLightGbmClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### GpuPythonSvmClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 7 imports
- **Logic/Math**: N/A

#### GpuPythonXGBoostClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 7 imports
- **Logic/Math**: N/A

#### GpuRandomForestClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 1 classes (e.g., ClassifierRegistry)
- **Dependencies**: 11 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.dataprovider
#### CsvDataProvider
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 3 classes (e.g., DashboardController, UploadViewController, AlgoSelectViewModel)
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### FastCsvParser
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### ImageDataProvider
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 11 imports
- **Logic/Math**: N/A

#### SyntheticDataGenerator
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., UploadViewController)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.gpu
#### ArrowTrainingDataSerializer
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 10 imports
- **Logic/Math**: N/A

#### GpuBatchService
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 2 classes (e.g., AbstractFeatureSelectionAlgorithm, PythonInferenceEngine)
- **Dependencies**: 15 imports
- **Logic/Math**: N/A

#### GpuJob
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### HardwareAcceleratorService
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 6 classes (e.g., AbstractFeatureSelectionAlgorithm, SystemMonitorService, ExecutionService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### IGpuCachedClassifier
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 6 classes (e.g., AbstractFeatureSelectionAlgorithm, GpuPythonCatBoostClassifier, GpuPythonLightGbmClassifier)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### ITrainingDataSerializer
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### PythonExecutionService
- **Status**: WIP (Contains TODOs)
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 15 classes (e.g., BertClassifier, CnnClassifier, CustomScriptClassifier)
- **Dependencies**: 10 imports
- **Logic/Math**: N/A

#### SerializedData
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: GPU Acceleration
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.lca
#### LcaService
- **Status**: Implemented
- **High Level Feature**: Autonomous Agent
- **Specific Contribution**: Layered Cognitive Agent (LCA)
- **Usage**: Used by 1 classes (e.g., AgentDebugViewController)
- **Dependencies**: 15 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.lca.actor
#### ActionDispatcher
- **Status**: Implemented
- **High Level Feature**: Autonomous Agent
- **Specific Contribution**: Layered Cognitive Agent (LCA)
- **Usage**: Used by 1 classes (e.g., LcaService)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.lca.perception
#### PerceptionService
- **Status**: Implemented
- **High Level Feature**: Autonomous Agent
- **Specific Contribution**: Layered Cognitive Agent (LCA)
- **Usage**: Used by 1 classes (e.g., LcaService)
- **Dependencies**: 14 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.model
#### AlgoSimulationData
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### AlgorithmRecommendation
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., ExecutionService, RecommendationService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### AttributeSummary
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., RecommendationService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### ClassifierSimulationData
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### Dataset
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 30 classes (e.g., AbstractFeatureSelectionAlgorithm, AtomSearchOptimization, BinaryParticleSwarmOptimization)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### DatasetMetaFeatures
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., RecommendationService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### ExecutionResult
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 8 classes (e.g., AbstractFeatureSelectionAlgorithm, ManualSelectionAlgorithm, OptimizationAlgorithm)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### PreprocessingConfig
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### TransformationPreview
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### TuningSimulationData
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.model.hyperparameter
#### Hyperparameter
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### HyperparameterSpace
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.model.progress
#### AlgoProgress
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### IterationSample
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### OrchestratorState
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., ProgressViewController)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### Phase
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.registry
#### AlgorithmRegistry
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., RecommendationService)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### ClassifierRegistry
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., EmpiricalBenchmarker, RecommendationService)
- **Dependencies**: 18 imports
- **Logic/Math**: N/A

#### EnvironmentRegistry
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 8 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.rl
#### IEnvironment
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 4 classes (e.g., EnvironmentRegistry, GridWorldEnv, RlLabViewController)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### RlTrainingService
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., RlLabViewController, RlVisualizerController)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### TicTacToeEnv
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., RlLabViewController)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.rl.agent
#### IRLAgent
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., RlTrainingService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### TabularQLearningAgent
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., RlTrainingService, InsightExtractionService)
- **Dependencies**: 7 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.rl.extensions
#### GridWorldEnv
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 2 classes (e.g., EnvironmentRegistry, RlVisualizerController)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.service
#### AppStateService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 13 classes (e.g., AbstractFeatureSelectionAlgorithm, ManualSelectionAlgorithm, GpuBatchService)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### DynamicCompilerService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., CreatorLabViewController)
- **Dependencies**: 8 imports
- **Logic/Math**: N/A

#### IClassifier
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 41 classes (e.g., AblationAwareGeneticAlgorithm, AbstractFeatureSelectionAlgorithm, AtomSearchOptimization)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### LibraryManagerService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., ModelHubViewController)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### LogService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 58 classes (e.g., BinaryParticleSwarmOptimization, DifferentialEvolution, GeneticAlgorithm)
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### OllamaService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 2 classes (e.g., SotaLookupService, ChatViewController)
- **Dependencies**: 17 imports
- **Logic/Math**: N/A

#### PlatformService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., PythonExecutionService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### RealTimeDataService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., DataVisualizationViewController)
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### SotaLookupService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., AgenticAdvisor)
- **Dependencies**: 7 imports
- **Logic/Math**: N/A

#### StatusDumperService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 11 imports
- **Logic/Math**: N/A

#### TelemetryService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., GpuBatchService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.service.agentic
#### ActiveExplorationService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 2 classes (e.g., RlLabViewController, RlVisualizerController)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### AgenticAdvisor
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 7 imports
- **Logic/Math**: N/A

#### AgenticArchitectService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., CreatorLabViewController)
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### AutoAblationService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### BehavioralCloningService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 3 classes (e.g., GlobalInputListener, RlLabViewController, RlVisualizerController)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### CodeModificationService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### CodeStructureService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 8 imports
- **Logic/Math**: N/A

#### ConstitutionalService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 2 classes (e.g., OllamaService, ChatViewController)
- **Dependencies**: 10 imports
- **Logic/Math**: N/A

#### DomainGuardService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### EmpiricalBenchmarker
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### ErrorMonitorService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### ExpertStrategyService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., RlVisualizerController)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### GameKnowledgeService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### InputSimService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### InsightExtractionService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., RlVisualizerController)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### IntrospectionService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., RlVisualizerController)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### KnowledgeBaseService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., RlVisualizerController)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### MetaFeatureExtractor
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### MetaFeatures
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### PerformanceHistoryService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., PerformanceHistoryViewController)
- **Dependencies**: 8 imports
- **Logic/Math**: N/A

#### PipelineSuggestion
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### ProjectContextService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., OllamaService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### SLAMService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### ScreenWatcherService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., NeuralForgeViewController)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### SelfCorrectionLoopService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### StrategyInjectorService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., RlVisualizerController)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### SystemMonitorService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### TestCodeStructure
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### TestDomainGuard
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### VisionStateDetector
- **Status**: WIP (Contains TODOs)
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### WindowFocusService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 4 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.service.agentic.inference
#### IInferenceEngine
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 6 classes (e.g., GpuPythonCatBoostClassifier, GpuPythonLightGbmClassifier, GpuPythonSvmClassifier)
- **Dependencies**: 1 imports
- **Logic/Math**: N/A

#### OnnxExportService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### OnnxInferenceEngine
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 6 classes (e.g., GpuPythonCatBoostClassifier, GpuPythonLightGbmClassifier, GpuPythonSvmClassifier)
- **Dependencies**: 10 imports
- **Logic/Math**: N/A

#### PythonInferenceEngine
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 6 classes (e.g., GpuPythonCatBoostClassifier, GpuPythonLightGbmClassifier, GpuPythonSvmClassifier)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.service.pipeline
#### ArrowService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 16 imports
- **Logic/Math**: N/A

#### EtaEstimator
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., ProgressTrackingService)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A

#### ExecutionService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., DashboardController)
- **Dependencies**: 25 imports
- **Logic/Math**: N/A

#### HyperparameterTuningService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., DashboardController)
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### OnnxInferenceService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### PipelineValidator
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., PipelineSimulatorController)
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### PreprocessorService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., DashboardController)
- **Dependencies**: 3 imports
- **Logic/Math**: N/A

#### RecommendationService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 17 imports
- **Logic/Math**: N/A

#### SimulationService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 1 classes (e.g., PipelineSimulatorController)
- **Dependencies**: 2 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.service.progress
#### ProgressListener
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### ProgressTrackingService
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 5 classes (e.g., AbstractFeatureSelectionAlgorithm, StatusDumperService, SimulationService)
- **Dependencies**: 6 imports
- **Logic/Math**: N/A

#### ProgressUpdate
- **Status**: Implemented
- **High Level Feature**: System Orchestration
- **Specific Contribution**: Backend Service
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

## Module: desktop-app

### Package: com.example.automl_prototype_1
#### DashboardApplication
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 1 classes (e.g., SettingsViewController)
- **Dependencies**: 7 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.controller
#### AgentDebugViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### AlgoSelectViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### ChatViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 12 imports
- **Logic/Math**: N/A

#### Cleanable
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 0 imports
- **Logic/Math**: N/A

#### CreatorLabViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 8 imports
- **Logic/Math**: N/A

#### DashboardController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 30 imports
- **Logic/Math**: N/A

#### DataVisualizationViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 19 imports
- **Logic/Math**: N/A

#### FeatureSelectionDialogController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### GlobalInputListener
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 1 classes (e.g., RlLabViewController)
- **Dependencies**: 12 imports
- **Logic/Math**: N/A

#### ModelHubViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 9 imports
- **Logic/Math**: N/A

#### NeuralForgeViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 13 imports
- **Logic/Math**: N/A

#### PerformanceHistoryViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 8 imports
- **Logic/Math**: N/A

#### PipelineSimulatorController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 10 imports
- **Logic/Math**: N/A

#### ProgressViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 5 imports
- **Logic/Math**: N/A

#### ResultsViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 33 imports
- **Logic/Math**: N/A

#### RlLabViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 13 imports
- **Logic/Math**: N/A

#### RlVisualizerController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 28 imports
- **Logic/Math**: N/A

#### SettingsViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 4 imports
- **Logic/Math**: N/A

#### UploadViewController
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI Controller
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 16 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.util
#### LogCollector
- **Status**: Implemented
- **High Level Feature**: Infrastructure
- **Specific Contribution**: Start-up/Core Infrastructure
- **Usage**: Used by 0 classes (e.g., )
- **Dependencies**: 2 imports
- **Logic/Math**: N/A


### Package: com.example.automl_prototype_1.viewmodel
#### AlgoSelectViewModel
- **Status**: Implemented
- **High Level Feature**: User Interface
- **Specific Contribution**: UI View
- **Usage**: Used by 2 classes (e.g., AlgoSelectViewController, ModelHubViewController)
- **Dependencies**: 9 imports
- **Logic/Math**: N/A
