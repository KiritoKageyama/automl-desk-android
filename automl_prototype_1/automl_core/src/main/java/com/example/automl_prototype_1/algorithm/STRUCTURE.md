# Package: Algorithm

## Feature Context
This package houses the **Core Optimization and Machine Learning Algorithms**. It includes implementations of evolutionary algorithms (Genetic Algorithm, PSO, Firefly) and wrappers for standard ML classifiers (Random Forest, SVM, XGBoost). These define *how* the system learns.

## Architecture Position
**Compute / Model Layer**
- **Upstream**: Instantiated and run by `ExecutionService` or `HyperparameterTuningService`.
- **Downstream**: Consumes `Dataset` objects and produces `Model` or `ExecutionResult` objects.

## File Map
- **`GeneticAlgorithm.java`**: Implementation of standard GA for feature selection or hyperparameter tuning.
- **`SimulatedAnnealing.java`**: Probabilistic technique for approximating the global optimum of a given function.
- **`ParticleSwarmOptimization.java`**: Population-based metaheuristic optimization algorithm.
- **`MlpClassifier.java`**: Wrapper for a Multi-Layer Perceptron (Neural Network).
- **`AbstractAlgorithm.java`**: Base class defining the contract (train, predict, evaluate) for all algorithms.
