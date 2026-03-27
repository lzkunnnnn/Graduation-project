package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorEventDTO;
import org.example.transportation.dox.sensor.SensorFlow;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;


@Repository
public interface SensorFlowRepository extends CrudRepository<SensorFlow, String> {

    @Query("select name,address from sensor_flow")
    List<SensorEventDTO> findAllSensorFlow();

    @Query("select * from sensor_flow ")
    List<SensorFlow> findAll();

    @Query("select count(*) from sensor_flow")
    int countNum();

    //state为1代表正常
    @Query("select count(*) from sensor_flow where state!=1")
    int countAbnormalNum();

    @Query("select president from sensor_flow where name=:name")
    String findPresidentByName(@Param("name") String name);

}
