package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorEventDTO;
import org.example.transportation.dox.sensor.SensorFlow;
import org.example.transportation.dox.sensor.SensorLane;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SensorLaneRepository extends CrudRepository<SensorLane, String> {

    @Query("select name,address from sensor_lane")
    List<SensorEventDTO> findAllSensorLane();

    @Query("select * from sensor_lane ")
    List<SensorLane> findAll();

    @Query("select count(*) from sensor_lane")
    int countNum();

    //state为1代表正常
    @Query("select count(*) from sensor_lane where state!=1")
    int countAbnormalNum();

    @Query("select president from sensor_lane where name=:name")
    String findPresidentByName(@Param("name") String name);
}
