package com.example.automl_prototype_1.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class HardwareAcceleratorService {

    private static java.util.Map<String, com.aparapi.device.Device> availableDevices = new java.util.HashMap<>();
    private static Boolean gpuAvailable = null;

    public static java.util.List<String> getAvailableGpuNames() {
        if (availableDevices.isEmpty()) {
            scanDevices();
        }
        return new java.util.ArrayList<>(availableDevices.keySet());
    }

    public static com.aparapi.device.Device getDeviceByName(String name) {
        if (availableDevices.isEmpty()) {
            scanDevices();
        }
        return availableDevices.get(name);
    }

    private static void scanDevices() {
        availableDevices.clear();
        try {
            // Scan for all OpenCL GPU devices
            java.util.List<com.aparapi.device.OpenCLDevice> devices = com.aparapi.device.OpenCLDevice
                    .listDevices(com.aparapi.device.Device.TYPE.GPU);

            for (com.aparapi.device.Device device : devices) {
                String name = device.getShortDescription();
                // Ensure unique names if multiple identical GPUs
                int counter = 2;
                String uniqueName = name;
                while (availableDevices.containsKey(uniqueName)) {
                    uniqueName = name + " (" + counter++ + ")";
                }
                availableDevices.put(uniqueName, device);
                System.out.println("HardwareAcceleratorService: Found GPU -> " + uniqueName);
            }
        } catch (Throwable t) {
            System.err.println("HardwareAcceleratorService: Error scanning devices: " + t.getMessage());
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
        if (!availableDevices.isEmpty()) {
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
        // If we have devices, return the first one as default
        if (!availableDevices.isEmpty()) {
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
}
