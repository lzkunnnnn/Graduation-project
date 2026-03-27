package org.example.transportation.repository;

import org.example.transportation.dox.event.Event;
import org.springframework.data.jdbc.repository.query.Modifying;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface EventRepository extends CrudRepository<Event, String> {

    @Query("select * from event")
    List<Event> findAllEvents();

    @Query("select * from event where id=:id")
    Event findEventById(@Param("id") int id);


    //上报人拓展
    @Query("select * from event where report_person=:reportPerson and id!=:id")
    List<Event> findEventsByReportPerson(@Param("reportPerson") String reportPerson, @Param("id") int id);


    //监控拓展
    @Query("select name from sensor_speed where address=:address")
    List<String> findSpeedSensorsByAddress(@Param("address") String address);
    @Query("select name from sensor_park where address=:address")
    List<String> findParkSensorsByAddress(@Param("address") String address);
    @Query("select name from sensor_flow where address=:address")
    List<String> findFlowSensorsByAddress(@Param("address") String address);
    @Query("select name from sensor_check where address=:address")
    List<String> findCheckSensorsByAddress(@Param("address") String address);
    @Query("select name from sensor_lane where address=:address")
    List<String> findLaneSensorsByAddress(@Param("address") String address);
    @Query("select name from sensor_environment where address=:address")
    List<String> findEnvironmentSensorsByAddress(@Param("address") String address);


    @Query("select count(*) from event where state=0 and level=:level")
    int countHandledEvents(int level);

    @Query("select count(*) from event where state=1 and level=:level")
    int countUnhandledEvents(int level);

    @Query("select count(*) from event where state=0")
    int countHandledNum();

    @Query("select count(*) from event where state =1")
    int countUnHandledNum();


    //处理/未处理状态转变
    @Query("update event set state=0,handle_time = current_timestamp,handle_person=:handlePerson where id = :id")
    @Modifying
    void handleEvent(int id,String handlePerson);

    @Query("update event set state=1,handle_time=null,handle_person=null where id =:id")
    @Modifying
    void unHandleEvent(int id,String reportPerson);

    //统计报警类型占比
    @Query("select count(*) from event where type=:type")
    int countTypeNum(String type);

    @Query("update sensor_flow set name=:name where id=:id")
    @Modifying
    void updateWater(@Param("name") String name,int id);


    @Query("select count(*) from sensor_park")
    int countRows();

    @Query("update sensor_park set state=:state,level=:level where id=:id")
    @Modifying
    void updateSL(int state,int level,int id);
}
