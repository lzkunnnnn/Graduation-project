package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorCheck;
import org.example.transportation.dox.sensor.SensorEventDTO;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SensorCheckRepository extends CrudRepository<SensorCheck, String> {

    @Query("select name,address from transport.sensor_check")
    List<SensorEventDTO> findAllSensorCheck();

    @Query("select * from sensor_check")
    List<SensorCheck> findAll();

    @Query("select count(*) from sensor_check")
    int countNum();

    @Query("select count(*) from sensor_check where state!=1")
    int countAbnormalNum();

}
