package com.example.automl_prototype_1.gpu;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class HardwareAcceleratorService {

    // Lazy initialization to prevent static block failures if Aparapi is missing
    // from classpath
    private static java.util.Map<String, com.aparapi.device.Device> availableDevices = null;
    private static Boolean gpuAvailable = null;

    public static java.util.List<String> getAvailableGpuNames() {
        if (availableDevices == null) {
            scanDevices();
        }
        return new java.util.ArrayList<>(availableDevices.keySet());
    }

    public static com.aparapi.device.Device getDeviceByName(String name) {
        if (availableDevices == null) {
            scanDevices();
        }
        return availableDevices != null ? availableDevices.get(name) : null;
    }

    private static void scanDevices() {
        if (availableDevices == null) {
            availableDevices = new java.util.HashMap<>();
        } else {
            availableDevices.clear();
        }

        try {
            // Scan for all OpenCL GPU devices
            // We wrap this specifically to catch NoClassDefFoundError if Aparapi is missing
            java.util.List<com.aparapi.device.OpenCLDevice> devices = com.aparapi.device.OpenCLDevice
                    .listDevices(com.aparapi.device.Device.TYPE.GPU);

            for (com.aparapi.device.Device device : devices) {
                String name = device.getShortDescription();
                // Simplification for User Experience:
                // Deduplicate by name. If multiple OpenCL platforms (e.g. generic vs vendor)
                // detect the same GPU, we don't want to list it twice.
                // If the user effectively has two identical cards, they'll see one entry, which
                // is acceptable for this use case.
                availableDevices.put(name, device);
                System.out.println("HardwareAcceleratorService: Found GPU -> " + name);
            }
        } catch (Throwable t) {
            // Catching Throwable includes NoClassDefFoundError
            System.err.println(
                    "HardwareAcceleratorService: Error scanning devices (Aparapi might be missing): " + t.getMessage());
        }

        // Also add logic to check default if list is empty (fallback)
        if (availableDevices.isEmpty()) {
            checkAparapiGpu(); // populate detectedGpuName legacy way
            if (gpuAvailable != null && gpuAvailable) {
                // We found something via simple check, but listDevices failed?
                // Just leave map empty or try to add "Default GPU"
            }
        }
    }

    /**
     * Checks if a GPU is available on the system.
     */
    public static boolean isGpuAvailable() {
        // If we already checked and found it available
        if (gpuAvailable != null && gpuAvailable) {
            return true;
        }
        // If we already checked and it was false, we might want to retry implicitly by
        // falling through,
        // or just return false. For now, let's allow re-scan if devices are null.
        if (availableDevices != null && !availableDevices.isEmpty()) {
            gpuAvailable = true;
            return true;
        }

        // Force scan
        scanDevices();
        if (!availableDevices.isEmpty()) {
            gpuAvailable = true;
            return true;
        }

        // Fallback checks
        return checkAparapiGpu() || checkSystemGpuFallback();
    }

    @SuppressWarnings("deprecation")
    private static boolean checkAparapiGpu() {
        try {
            com.aparapi.device.Device device = com.aparapi.device.Device.best();
            if (device != null && device.getType() == com.aparapi.device.Device.TYPE.GPU) {
                detectedGpuName = device.getShortDescription();
                return true;
            }
        } catch (Throwable t) {
            System.err.println("Aparapi check failed: " + t.getMessage());
        }
        return false;
    }

    private static boolean checkSystemGpuFallback() {
        String os = System.getProperty("os.name", "generic").toLowerCase(java.util.Locale.ENGLISH);
        if (os.contains("win"))
            return checkWindowsGpu();
        if (os.contains("nux"))
            return checkLinuxGpu();
        if (os.contains("mac"))
            return true;
        return false;
    }

    private static String detectedGpuName = "Unknown";

    public static String getGpuName() {
        if (gpuAvailable == null)
            isGpuAvailable();

        // Prioritize NVIDIA if multiple GPUs exist
        if (availableDevices != null && !availableDevices.isEmpty()) {
            for (String name : availableDevices.keySet()) {
                if (name.toLowerCase().contains("nvidia")) {
                    return name;
                }
            }
            // Fallback: Return the first one found
            return availableDevices.keySet().iterator().next();
        }
        return detectedGpuName;
    }

    // Keep OS specific checks as fallback
    private static boolean checkWindowsGpu() {
        try {
            ProcessBuilder pb = new ProcessBuilder("powershell", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name");
            Process p = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                String normalized = line.toLowerCase();
                if (normalized.contains("nvidia") || normalized.contains("amd") || normalized.contains("radeon")
                        || normalized.contains("geforce")) {
                    detectedGpuName = line.trim();
                    return true;
                }
            }
        } catch (Exception e) {
        }
        return false;
    }

    private static boolean checkLinuxGpu() {
        try {
            ProcessBuilder pb = new ProcessBuilder("lspci");
            Process p = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                String normalized = line.toLowerCase();
                if ((normalized.contains("vga") || normalized.contains("3d"))
                        && (normalized.contains("nvidia") || normalized.contains("amd"))) {
                    detectedGpuName = line.trim();
                    return true;
                }
            }
        } catch (Exception e) {
        }
        return false;
    }

    /**
     * Calculates the optimal number of parallel threads based on available CPU
     * cores
     * and memory.
     * Strategy:
     * 1. Check available processor cores.
     * 2. Check available physical memory (via OperatingSystemMXBean).
     * 3. Target using roughly 50% of available free resources.
     * 4. Ensure at least 1 thread.
     */
    @SuppressWarnings("deprecation")
    public static int calculateOptimalThreadCount() {
        int availableCores = Runtime.getRuntime().availableProcessors();
        long freeMemoryBytes = Runtime.getRuntime().freeMemory(); // Fallback if OSBean fails

        try {
            java.lang.management.OperatingSystemMXBean osBean = java.lang.management.ManagementFactory
                    .getOperatingSystemMXBean();
            if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
                com.sun.management.OperatingSystemMXBean nativeOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
                long physicalFree = nativeOsBean.getFreePhysicalMemorySize();
                if (physicalFree > 0) {
                    freeMemoryBytes = physicalFree;
                }
            }
        } catch (Throwable t) {
            // Ignore if restricted or not available
        }

        // HEURISTIC:
        // Each Python/GPU worker needs roughly 1GB RAM overhead (conservative
        // estimate).
        // We want to use half of the *available* cores, or 1 if very low.

        // HEURISTIC:
        // Adjusted: Assume ~100MB overhead per thread (sufficient for CPU tasks and
        // batched GPU calls)
        int memoryConstrainedLimit = (int) (freeMemoryBytes / (100L * 1024L * 1024L));
        if (memoryConstrainedLimit < 1)
            memoryConstrainedLimit = 1;

        // Target using most cores but leave some for UI/OS
        int coreTarget = availableCores - 2;
        if (coreTarget < 1)
            coreTarget = 1;

        // Take the smaller of limit by memory vs limit by cores
        int optimal = Math.min(coreTarget, memoryConstrainedLimit);

        // Cap max to avoid excessive context switching overhead, but allow more than 12
        if (optimal > 24)
            optimal = 24;

        // Phase 6 Step 3: Explicit Resource Cap for 16GB Systems
        // If Total System Memory is <= 16GB, cap threads strictly to 4 to leave room
        // for Python/OS
        long totalMemory = -1;
        try {
            java.lang.management.OperatingSystemMXBean osBean = java.lang.management.ManagementFactory
                    .getOperatingSystemMXBean();
            if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
                totalMemory = ((com.sun.management.OperatingSystemMXBean) osBean).getTotalPhysicalMemorySize();
            }
        } catch (Exception e) {
        }

        if (totalMemory > 0 && totalMemory <= 17L * 1024 * 1024 * 1024) { // <= 17GB (approx 16GB + margin)
            if (optimal > 4) {
                System.out.println("HardwareAccelerator: 16GB System Detected. Capping threads to 4 for safety.");
                optimal = 4;
            }
        }

        // System.out.println("HardwareAccelerator: Optimal Threads = " + optimal +
        // " (Cores=" + availableCores + ", FreeRAM=" + (freeMemoryBytes / 1024 / 1024)
        // + "MB)");

        return optimal;
    }
}
