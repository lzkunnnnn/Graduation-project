package org.example.transportation.service;

import lombok.RequiredArgsConstructor;
import org.example.transportation.dox.sensor.*;
import org.example.transportation.repository.*;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class SensorService {
    private final SensorFlowRepository sensorFlowRepository;
    private final SensorParkRepository sensorParkRepository;
    private final SensorCheckRepository sensorCheckRepository;
    private final SensorSpeedRepository sensorSpeedRepository;
    private final SensorEnvironmentRepository environmentRepository;
    private final SensorLaneRepository sensorLaneRepository;

    public List<SensorEventDTO> getSensors(String sensorType){
        if(sensorType.equals("check")){
            return sensorCheckRepository.findAllSensorCheck();
        }
        else if(sensorType.equals("speed")){
            return sensorSpeedRepository.findAllSensorSpeed();
        }
        else if(sensorType.equals("park")){
            return sensorParkRepository.findAllSensorPark();
        }
        else if(sensorType.equals("flow")){
            return sensorFlowRepository.findAllSensorFlow();
        }
        else if(sensorType.equals("environment")){
            return environmentRepository.findAllSensorEnvironment();
        }
        else if(sensorType.equals("lane")){
            return sensorLaneRepository.findAllSensorLane();
        }
        return null;
    }



    public SensorDTO getDeviceState(){
        SensorDTO sensorDTO = SensorDTO.builder()
                .flowNum(sensorFlowRepository.countNum())
                .flowNumAbnormal(sensorFlowRepository.countAbnormalNum())
                .speedNum(sensorSpeedRepository.countNum())
                .speedNumAbnormal(sensorSpeedRepository.countAbnormalNum())
                .parkNum(sensorParkRepository.countNum())
                .parkNumAbnormal(sensorParkRepository.countAbnormalNum())
                .checkNum(sensorCheckRepository.countNum())
                .checkNumAbnormal(sensorCheckRepository.countAbnormalNum())
                .laneNum(sensorLaneRepository.countNum())
                .laneNumAbnormal(sensorLaneRepository.countAbnormalNum())
                .environmentNum(environmentRepository.countNum())
                .environmentNumAbnormal(environmentRepository.countAbnormalNum())
                .build();

        sensorDTO.setSensorNum(sensorDTO.getSpeedNum()
                            +sensorDTO.getFlowNum()
                            +sensorDTO.getParkNum()
                            +sensorDTO.getCheckNum()
                            +sensorDTO.getLaneNum()
                            +sensorDTO.getEnvironmentNum());

        sensorDTO.setAbnormalSensorNum(sensorDTO.getParkNumAbnormal()
                                        +sensorDTO.getCheckNumAbnormal()
                                        +sensorDTO.getFlowNumAbnormal()
                                        +sensorDTO.getSpeedNumAbnormal()
                                        +sensorDTO.getEnvironmentNumAbnormal()
                                        +sensorDTO.getLaneNumAbnormal());

        sensorDTO.setOnlineSensorNum(sensorDTO.getSensorNum()-sensorDTO.getAbnormalSensorNum());

        return sensorDTO;
    }

    //flow
    public List<SensorFlow> findFlowAll(){
        return sensorFlowRepository.findAll();
    }

    public int countFlowNum(){
        return sensorFlowRepository.countNum();
    }

    public int countFlowAbnormalNum(){
        return sensorFlowRepository.countAbnormalNum();
    }
    //park
    public List<SensorPark> findParkAll(){
        return sensorParkRepository.findAll();
    }

    public int countParkNum(){
        return sensorParkRepository.countNum();
    }

    public int countParkAbnormalNum(){
        return sensorParkRepository.countAbnormalNum();
    }
    //check
    public List<SensorCheck> findCheckAll(){
        return sensorCheckRepository.findAll();
    }

    public int countCheckNum(){  return sensorCheckRepository.countNum();    }

    public int countCheckAbnormalNum(){
        return sensorCheckRepository.countAbnormalNum();
    }
    //speed
    public List<SensorSpeed> findSpeedAll(){
        return sensorSpeedRepository.findAll();
    }

    public int countSpeedNum(){
        return sensorSpeedRepository.countNum();
    }

    public int countSpeedAbnormalNum(){
        return sensorSpeedRepository.countAbnormalNum();
    }

    //lane
    public List<SensorLane> findLaneAll(){
        return sensorLaneRepository.findAll();
    }
    public int countLaneNum(){
        return sensorLaneRepository.countNum();
    }
    public int countLaneAbnormalNum(){
        return sensorLaneRepository.countAbnormalNum();
    }
    //environment
    public List<SensorEnvironment> findEnvironmentAll(){
        return environmentRepository.findAll();
    }
    public int countEnvironmentNum(){
        return environmentRepository.countNum();
    }
    public int countEnvironmentAbnormalNum(){
        return environmentRepository.countAbnormalNum();
    }

}
