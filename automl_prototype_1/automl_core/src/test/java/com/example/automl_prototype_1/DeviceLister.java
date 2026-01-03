package com.example.automl_prototype_1;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import java.util.List;

public class DeviceLister {
    public static void main(String[] args) {
        System.out.println("--- Aparapi Device Lister ---");
        try {
            List<OpenCLDevice> devices = OpenCLDevice.listDevices(Device.TYPE.GPU);
            System.out.println("Found " + devices.size() + " GPU devices.");
            for (Device device : devices) {
                System.out.println("Device: " + device.getShortDescription() + " | Type: " + device.getType());
                System.out.println("   ID: " + device.getDeviceId());
            }
        } catch (Throwable t) {
            System.err.println("Error listing devices: " + t.getMessage());
            t.printStackTrace();
        }
    }
}
