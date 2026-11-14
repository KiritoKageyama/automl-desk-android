package com.example.automl_prototype_1.gpu;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ScriptAllowlistTest {

    @Test
    void rejectsArbitraryPaths() {
        assertFalse(ScriptAllowlist.isAllowed("C:/Windows/System32/evil.py"));
        assertFalse(ScriptAllowlist.isAllowed(null));
        assertFalse(ScriptAllowlist.isAllowed(""));
        assertFalse(ScriptAllowlist.isAllowed("not-python.txt"));
    }

    @Test
    void allowsUnderAutomlScripts() throws Exception {
        Path dir = Path.of(System.getProperty("user.home"), ".automl", "scripts");
        Files.createDirectories(dir);
        Path script = dir.resolve("unit_allowlist_test.py");
        Files.writeString(script, "def train(X,y,params): return None\n");
        assertTrue(ScriptAllowlist.isAllowed(script.toString()));
    }
}
