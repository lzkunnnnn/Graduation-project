package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorEventDTO;
import org.example.transportation.dox.sensor.SensorSpeed;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SensorSpeedRepository extends CrudRepository<SensorSpeed, String> {

    @Query("select name,address from sensor_speed")
    List<SensorEventDTO> findAllSensorSpeed();

    @Query("select * from sensor_speed")
    List<SensorSpeed> findAll();

    @Query("select count(*) from sensor_speed")
    int countNum();

    @Query("select count(*) from sensor_speed where state!=1")
    int countAbnormalNum();

}
