package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.gpu.HardwareAcceleratorService;
import com.example.automl_prototype_1.service.LogService;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;

/**
 * Monitors system resources (RAM, CPU, GPU) to provide context for the
 * Constitution.
 */
@SuppressWarnings("deprecation")
public class SystemMonitorService {

    private final OperatingSystemMXBean osBean;
    private final com.sun.management.OperatingSystemMXBean nativeOsBean;

    public SystemMonitorService() {
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
            this.nativeOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
        } else {
            this.nativeOsBean = null;
            LogService.warn("SystemMonitorService", "Native OS Bean not available. RAM metrics may be limited.");
        }
    }

    /**
     * @return Free Physical RAM in GB (approx)
     */
    public double getFreeRamGb() {
        if (nativeOsBean != null) {
            long bytes = nativeOsBean.getFreePhysicalMemorySize();
            return bytes / (1024.0 * 1024.0 * 1024.0);
        }
        return Runtime.getRuntime().freeMemory() / (1024.0 * 1024.0 * 1024.0); // Fallback to JVM free memory
    }

    /**
     * @return Total Physical RAM in GB (approx)
     */
    public double getTotalRamGb() {
        if (nativeOsBean != null) {
            long bytes = nativeOsBean.getTotalPhysicalMemorySize();
            return bytes / (1024.0 * 1024.0 * 1024.0);
        }
        return -1.0;
    }

    /**
     * @return System CPU Load (0.0 to 1.0)
     */
    public double getCpuLoad() {
        if (nativeOsBean != null) {
            double load = nativeOsBean.getSystemCpuLoad();
            return load < 0 ? 0.0 : load; // handle negative return
        }
        return 0.5; // Assume half load if unknown
    }

    /**
     * @return True if a dedicated GPU is detected.
     */
    public boolean hasGpu() {
        return HardwareAcceleratorService.isGpuAvailable();
    }

    public String getSystemContextString() {
        return String.format("RAM: %.1f/%.1f GB | CPU: %.0f%% | GPU: %s",
                getFreeRamGb(), getTotalRamGb(), getCpuLoad() * 100, hasGpu() ? "Yes" : "No");
    }
}
