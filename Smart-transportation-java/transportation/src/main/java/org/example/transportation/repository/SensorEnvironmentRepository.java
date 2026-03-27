package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorEnvironment;
import org.example.transportation.dox.sensor.SensorEnvironment;
import org.example.transportation.dox.sensor.SensorEventDTO;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SensorEnvironmentRepository extends CrudRepository<SensorEnvironment, String> {

    @Query("select name,address from transport.sensor_environment")
    List<SensorEventDTO> findAllSensorEnvironment();

    @Query("select * from sensor_environment ")
    List<SensorEnvironment> findAll();

    @Query("select count(*) from sensor_environment")
    int countNum();

    //state为1代表正常
    @Query("select count(*) from sensor_environment where state!=1")
    int countAbnormalNum();

    @Query("select president from sensor_environment where name=:name")
    String findPresidentByName(@Param("name") String name);
}
