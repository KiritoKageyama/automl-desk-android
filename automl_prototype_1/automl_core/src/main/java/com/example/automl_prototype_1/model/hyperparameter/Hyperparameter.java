package com.example.automl_prototype_1.model.hyperparameter;

import java.util.List;

/**
 * Represents a single configurable hyperparameter.
 */
public class Hyperparameter {
    public enum Type {
        INTEGER, DOUBLE, CATEGORICAL, BOOLEAN
    }

    private final String name;
    private final Type type;
    private final Object defaultValue;
    private final Object min; // For numeric
    private final Object max; // For numeric
    private final List<Object> options; // For categorical

    // Constructor for Numeric (Int/Double)
    public Hyperparameter(String name, Type type, Object defaultValue, Object min, Object max) {
        this.name = name;
        this.type = type;
        this.defaultValue = defaultValue;
        this.min = min;
        this.max = max;
        this.options = null;
    }

    // Constructor for Categorical
    public Hyperparameter(String name, Object defaultValue, List<Object> options) {
        this.name = name;
        this.type = Type.CATEGORICAL;
        this.defaultValue = defaultValue;
        this.min = null;
        this.max = null;
        this.options = options;
    }

    // Constructor for Boolean
    public Hyperparameter(String name, Boolean defaultValue) {
        this.name = name;
        this.type = Type.BOOLEAN;
        this.defaultValue = defaultValue;
        this.min = null;
        this.max = null;
        this.options = null;
    }

    public String getName() {
        return name;
    }

    public Type getType() {
        return type;
    }

    public Object getDefaultValue() {
        return defaultValue;
    }

    public Object getMin() {
        return min;
    }

    public Object getMax() {
        return max;
    }

    public List<Object> getOptions() {
        return options;
    }
}
