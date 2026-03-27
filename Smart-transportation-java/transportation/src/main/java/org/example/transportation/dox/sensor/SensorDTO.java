package org.example.transportation.dox.sensor;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SensorDTO {
    private int sensorNum;
    private int onlineSensorNum;
    private int abnormalSensorNum;
    private int flowNum;
    private int flowNumAbnormal;
    private int parkNum;
    private int parkNumAbnormal;
    private int speedNum;
    private int speedNumAbnormal;
    private int checkNum;
    private int checkNumAbnormal;
    private int laneNum;
    private int laneNumAbnormal;
    private int environmentNum;
    private int environmentNumAbnormal;
}
