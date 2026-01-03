package com.example.automl_prototype_1.model.hyperparameter;

import java.util.List;

/**
 * Interface for components (Algorithms, Classifiers) that have tunable
 * hyperparameters.
 */
public interface HyperparameterSpace {
    /**
     * Returns the list of hyperparameters defined by this component.
     * 
     * @return List of Hyperparameter objects.
     */
    List<Hyperparameter> getHyperparameters();
}
