package org.example.transportation.controller;

import lombok.RequiredArgsConstructor;
import org.example.transportation.service.SensorService;
import org.example.transportation.vo.ResultVO;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/sensor/")
@RequiredArgsConstructor
public class SensorController {
    private final SensorService sensorService;

    @GetMapping("sensors")
    public ResultVO getSensors(@RequestParam("type")String type) {
        return ResultVO.success(sensorService.getSensors(type));
    }

    @Cacheable(value="deviceState",key="5")
    @GetMapping("deviceState")
    public ResultVO getDeviceState() {
        return ResultVO.success(sensorService.getDeviceState());
    }

    //flow
    @GetMapping("flow")
    public ResultVO findAllflow(){
        return ResultVO.success(sensorService.findFlowAll());
    }

//    @GetMapping("flow/count/all")
//    public ResultVO findAllflowCount(){
//        return  ResultVO.success(sensorService.countFlowNum());
//    }
//
//    @GetMapping("flow/count/abnormal")
//    public ResultVO findAllflowCountAbnormal(){
//        return  ResultVO.success(sensorService.countFlowAbnormalNum());
//    }

    //park
    @GetMapping("park")
    public ResultVO findAllPark(){
        return ResultVO.success(sensorService.findParkAll());
    }

//    @GetMapping("park/count/all")
//    public ResultVO findAllParkCount(){
//        return  ResultVO.success(sensorService.countParkNum());
//    }
//
//    @GetMapping("park/count/abnormal")
//    public ResultVO findAllParkCountAbnormal(){
//        return  ResultVO.success(sensorService.countParkAbnormalNum());
//    }

    //Speed
    @GetMapping("speed")
    public ResultVO findAllSpeed(){
        return ResultVO.success(sensorService.findSpeedAll());
    }

//    @GetMapping("speed/count/all")
//    public ResultVO findAllSpeedCount(){
//        return  ResultVO.success(sensorService.countSpeedNum());
//    }
//
//    @GetMapping("speed/count/abnormal")
//    public ResultVO findAllSpeedCountAbnormal(){
//        return  ResultVO.success(sensorService.countSpeedAbnormalNum());
//    }
    //check
    @GetMapping("check")
    public ResultVO findAllCheck(){
        return ResultVO.success(sensorService.findCheckAll());
    }
//    @GetMapping("check/count/all")
//    public ResultVO findAllCheckCount(){
//        return  ResultVO.success(sensorService.countCheckNum());
//    }
//    @GetMapping("check/count/abnormal")
//    public ResultVO findAllCheckCountAbnormal(){
//        return  ResultVO.success(sensorService.countCheckAbnormalNum());
//    }

    //lane
    @GetMapping("lane")
    public ResultVO findAllLane(){
        return ResultVO.success(sensorService.findLaneAll());
    }
//    @GetMapping("lane/count/all")
//    public ResultVO findAllLaneCount(){
//        return  ResultVO.success(sensorService.countLaneNum());
//    }
//    @GetMapping("lane/count/abnormal")
//    public ResultVO findAllLaneCountAbnormal(){
//        return  ResultVO.success(sensorService.countLaneAbnormalNum());
//    }

    //environment
    @GetMapping("environment")
    public ResultVO findAllEnvironment(){
        return ResultVO.success(sensorService.findEnvironmentAll());
    }
//    @GetMapping("environment/count/all")
//    public ResultVO findAllEnvironmentCount(){
//        return  ResultVO.success(sensorService.countEnvironmentNum());
//    }
//    @GetMapping("environment/count/abnormal")
//    public ResultVO findAllEnvironmentCountAbnormal(){
//        return  ResultVO.success(sensorService.countEnvironmentAbnormalNum());
//    }

}
