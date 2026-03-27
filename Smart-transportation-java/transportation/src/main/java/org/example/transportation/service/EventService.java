package org.example.transportation.service;


import lombok.RequiredArgsConstructor;

import org.example.transportation.dox.event.Event;
import org.example.transportation.dox.event.EventHandleDTO;
import org.example.transportation.repository.EventRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class EventService {
    private final EventRepository eventRepository;

    public List<Event> getEvents() {
        return eventRepository.findAllEvents();
    }

    public Event getEventById(int id) {
        return eventRepository.findEventById(id);
    }

    public int getHandledEventNum(int level){
        return eventRepository.countHandledEvents(level);
    }

    public int getUnhandledEventNum(int level){
        return eventRepository.countUnhandledEvents(level);
    }

    public int getTypeNum(String type){
        return eventRepository.countTypeNum(type);
    }

    public EventHandleDTO getEventHandleDTO() {

        List<Event> events = getEvents();
        List<Event> alarmEvents = events.stream().filter(e->e.getAlarmType().equals("警报")).collect(Collectors.toList());
        List<Event> earlyAlarmEvents = events.stream().filter(e->e.getAlarmType().equals("预警")).collect(Collectors.toList());

        return EventHandleDTO.builder()
                .alarmNum(alarmEvents.size())
                .earlyAlarmNum(earlyAlarmEvents.size())
                .checkNum(getTypeNum("check"))
                .flowNum(getTypeNum("flow"))
                .speedNum(getTypeNum("speed"))
                .parkNum(getTypeNum("park"))
                .environmentNum(getTypeNum("environment"))
                .laneNum(getTypeNum("lane"))
                .handledLevelOne(getHandledEventNum(0))
                .handledLevelTwo(getHandledEventNum(1))
                .handledLevelThree(getHandledEventNum(2))
                .handledLevelFour(getHandledEventNum(3))
                .unHandledLevelOne(getUnhandledEventNum(0))
                .unHandledLevelTwo(getUnhandledEventNum(1))
                .unHandledLevelThree(getUnhandledEventNum(2))
                .unHandledLevelFour(getUnhandledEventNum(3))
                .handledNum(eventRepository.countHandledNum())
                .unHandledNum(eventRepository.countUnHandledNum())
                .build();
    }

    public void changeEventState(int id,int state,String person){
        if(state==1){
            eventRepository.handleEvent(id,person);
        }
        else if(state==0){
            eventRepository.unHandleEvent(id,person);
        }
    }
    
    public void saveEvent(Event event){
        eventRepository.save(event);
    }





}
