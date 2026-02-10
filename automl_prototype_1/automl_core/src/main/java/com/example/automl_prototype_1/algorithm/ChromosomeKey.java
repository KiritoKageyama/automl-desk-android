package com.example.automl_prototype_1.algorithm;

import java.util.Arrays;

/**
 * Immutable key for caching chromosome fitness values.
 * replacing expensive Arrays.toString() keys.
 */
public final class ChromosomeKey {
    private final int[] selectedIndices;
    private final int hashCode;

    /**
     * Creates a key based on selected feature indices.
     * Values in the gene array > 0.5 are considered "selected" (1), others "not
     * selected" (0).
     *
     * @param genes The continuous-valued chromosome position.
     */
    public ChromosomeKey(double[] genes) {
        // Canonicalize: Convert continuous genes to selected indices
        // faster than BitSet for small-medium N (e.g. < 10k features), and safer for
        // array-based hashing.
        // NOTE: For very large feature sets (>10k), consider switching to BitSet or
        // MurmurHash
        // to avoid memory overhead and collision constraints.
        int[] temp = new int[genes.length];
        int count = 0;
        for (int i = 0; i < genes.length; i++) {
            if (genes[i] > 0.5) {
                temp[count++] = i;
            }
        }
        this.selectedIndices = Arrays.copyOf(temp, count);
        this.hashCode = Arrays.hashCode(this.selectedIndices);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        ChromosomeKey that = (ChromosomeKey) o;
        return Arrays.equals(selectedIndices, that.selectedIndices);
    }

    @Override
    public int hashCode() {
        return hashCode;
    }
}
