package com.example.automl_prototype_1.controller;

/**
 * An interface for controllers that need to perform cleanup
 * operations (e.g., stopping background threads) before
 * they are removed from the view.
 */
public interface Cleanable {
    void cleanup();
}