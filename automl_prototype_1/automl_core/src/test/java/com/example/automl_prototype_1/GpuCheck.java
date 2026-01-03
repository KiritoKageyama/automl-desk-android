
package com.example.automl_prototype_1;

import com.example.automl_prototype_1.service.HardwareAcceleratorService;
import org.junit.jupiter.api.Test;

public class GpuCheck {

    @Test
    public void testGpuDetection() {
        System.out.println("--- GPU Verification Check ---");
        boolean available = HardwareAcceleratorService.isGpuAvailable();
        System.out.println("GPU Available: " + available);

        if (available) {
            String name = HardwareAcceleratorService.getGpuName();
            System.out.println("Detected Device: " + name);
            System.out.println("Status: ACTIVE and READY for Acceleration.");
        } else {
            System.out.println("Status: CPU Only (No compatible discrete GPU detected via Aparapi).");
        }
        System.out.println("------------------------------");
    }
}
