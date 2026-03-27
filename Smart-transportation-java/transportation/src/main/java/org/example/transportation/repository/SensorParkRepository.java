package org.example.transportation.repository;

import org.example.transportation.dox.sensor.SensorEventDTO;
import org.example.transportation.dox.sensor.SensorPark;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository
public interface SensorParkRepository extends CrudRepository<SensorPark, String> {

    @Query("select name,address from sensor_park")
    List<SensorEventDTO> findAllSensorPark();

    @Query("select * from sensor_park  ")
    List<SensorPark> findAll();

    @Query("select count(*) from sensor_park")
    int countNum();

    @Query("select count(*) from sensor_park where state!=1")
    int countAbnormalNum();

}
