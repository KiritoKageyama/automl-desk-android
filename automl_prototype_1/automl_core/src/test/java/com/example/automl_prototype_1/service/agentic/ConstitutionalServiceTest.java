package com.example.automl_prototype_1.service.agentic;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ConstitutionalService — the AI safety code auditor.
 * Validates that dangerous patterns are caught and safe patterns are allowed.
 */
class ConstitutionalServiceTest {

    private ConstitutionalService service;

    @BeforeEach
    void setUp() {
        service = ConstitutionalService.getInstance();
    }

    // ── Singleton ──────────────────────────────────────────────────────────────

    @Test
    void getInstance_returnsSameInstance() {
        assertSame(ConstitutionalService.getInstance(), ConstitutionalService.getInstance());
    }

    // ── auditCode: safe code ──────────────────────────────────────────────────

    @Test
    void auditCode_safeClassifierCode_allowed() {
        String safeCode = """
                package com.example.automl_prototype_1.classifier;
                import com.example.automl_prototype_1.service.IClassifier;
                public class SafeClassifier implements IClassifier {
                    @Override public void train(double[][] features, int[] labels) {}
                    @Override public int[] predict(double[][] features) { return new int[0]; }
                    @Override public String getName() { return "Safe"; }
                }
                """;
        ConstitutionalService.Verdict v = service.auditCode(safeCode);
        assertTrue(v.allowed, "Safe classifier code should be allowed");
    }

    // ── auditCode: dangerous patterns ─────────────────────────────────────────

    @Test
    void auditCode_systemExit_vetoed() {
        String code = "public class Evil { void foo() { System.exit(0); } }";
        ConstitutionalService.Verdict v = service.auditCode(code);
        assertFalse(v.allowed, "System.exit() should be vetoed");
        assertNotNull(v.reason, "Veto should have a reason");
    }

    @Test
    void auditCode_runtimeExec_vetoed() {
        String code = "public class Evil { void foo() throws Exception { Runtime.getRuntime().exec(\"rm -rf /\"); } }";
        ConstitutionalService.Verdict v = service.auditCode(code);
        assertFalse(v.allowed, "Runtime.exec() should be vetoed");
    }

    @Test
    void auditCode_processBuilder_vetoed() {
        String code = "public class Evil { void foo() throws Exception { new ProcessBuilder(\"cmd\").start(); } }";
        ConstitutionalService.Verdict v = service.auditCode(code);
        assertFalse(v.allowed, "ProcessBuilder should be vetoed");
    }

    @Test
    void auditCode_fileDelete_vetoed() {
        String code = "import java.io.File; public class Evil { void foo() { new File(\"x\").delete(); } }";
        ConstitutionalService.Verdict v = service.auditCode(code);
        assertFalse(v.allowed, "File.delete() should be vetoed");
    }

    @Test
    void auditCode_reflection_vetoed() {
        String code = "public class Evil { void foo() throws Exception { Class.forName(\"com.evil\").newInstance(); } }";
        ConstitutionalService.Verdict v = service.auditCode(code);
        assertFalse(v.allowed, "Reflection should be vetoed");
    }

    // ── auditCode: edge cases ─────────────────────────────────────────────────

    @Test
    void auditCode_null_vetoed() {
        ConstitutionalService.Verdict v = service.auditCode(null);
        assertFalse(v.allowed, "Null code should be vetoed");
    }

    @Test
    void auditCode_empty_vetoed() {
        ConstitutionalService.Verdict v = service.auditCode("");
        assertFalse(v.allowed, "Empty code should be vetoed");
    }

    // ── Verdict ───────────────────────────────────────────────────────────────

    @Test
    void verdict_allow_hasCorrectState() {
        ConstitutionalService.Verdict v = ConstitutionalService.Verdict.allow();
        assertTrue(v.allowed);
        assertNull(v.reason);
    }

    @Test
    void verdict_veto_hasCorrectState() {
        ConstitutionalService.Verdict v = ConstitutionalService.Verdict.veto("test reason");
        assertFalse(v.allowed);
        assertEquals("test reason", v.reason);
    }

    // ── Strict mode ───────────────────────────────────────────────────────────

    @Test
    void setStrictMode_doesNotThrow() {
        assertDoesNotThrow(() -> service.setStrictMode(true));
        assertDoesNotThrow(() -> service.setStrictMode(false));
    }
}
