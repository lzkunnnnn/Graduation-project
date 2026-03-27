package org.example.transportation.controller;



import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.example.transportation.dox.event.Event;
import org.example.transportation.service.EventService;
import org.example.transportation.vo.ResultVO;
import org.springframework.cache.annotation.CachePut;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/event/")
@Slf4j
public class EventController {
    private final EventService eventService;

    @GetMapping("getEvents")
    public ResultVO getEvents() {
        return ResultVO.success(eventService.getEvents());
    }

    @GetMapping("getEvent")
    public ResultVO getEvent(@RequestParam int id) {
        return ResultVO.success(eventService.getEventById(id));
    }

    @GetMapping("getHandledEventsNum")
    public ResultVO getHandledEventsNum(int level) {
        return ResultVO.success(eventService.getHandledEventNum(level));
    }


    @GetMapping("getUnhandledEventsNum")
    public ResultVO getUnhandledEventsNum(int level) {
        return ResultVO.success(eventService.getUnhandledEventNum(level));
    }


    @GetMapping("getHandleDTO")
    public ResultVO getHandleDTO() {
        return ResultVO.success(eventService.getEventHandleDTO());
    }


    @GetMapping("handleEvent")
    public ResultVO handleEvent(@RequestParam int id,@RequestParam int state,@RequestParam String person){
        eventService.changeEventState(id,state,person);
        return ResultVO.success("success");
    }
    /*
    @PostMapping("handleEvent")
    public ResultVO handleEvent( @RequestBody EventDTO event){
        try{
            eventService.changeEventStateDTO(event);
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        return ResultVO.success("success");
    }
    */
    @GetMapping("addEvent")
    public ResultVO addEvent(@RequestParam("title")String title,@RequestParam("type")String type,@RequestParam("alarmType")String alarmType,
                             @RequestParam("address")String address,@RequestParam("content")String content,@RequestParam("source")String source,
                             @RequestParam("reportPerson")String reportPerson,@RequestParam("level")int level) {
        Event event = Event.builder().title(title).type(type).alarmType(alarmType)
                .address(address).content(content).source(source).reportPerson(reportPerson).level(level).state(1).build();
        eventService.saveEvent(event);
        return ResultVO.success(event);
    }

}
