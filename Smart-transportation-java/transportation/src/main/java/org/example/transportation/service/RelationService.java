package org.example.transportation.service;

import lombok.RequiredArgsConstructor;
import org.example.transportation.dox.event.Event;
import org.example.transportation.dox.relation.*;
import org.example.transportation.repository.EventRepository;

import org.springframework.stereotype.Service;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Service
@RequiredArgsConstructor
public class RelationService {

    private final EventRepository eventRepository;

    private static boolean isIgnored(String fieldName, String[] ignoredFields) {
        for (String ignoredField : ignoredFields) {
            if (fieldName.equals(ignoredField)) {
                return true;
            }
        }
        return false;
    }

    public String levelTransfor(int level){
        if(level==0)return "一级警报";
        if(level==1)return "二级警报";
        if(level==2)return "三级警报";
        if(level==3)return "四级警报";
        return "未知";
    }

    public ResultDTO getRelation(int id){
        Event event = eventRepository.findEventById(id);

        List<Link> links = new ArrayList<>();
        List<Item> items = new ArrayList<>();

        String title = event.getTitle();
        String address = event.getAddress();
        String source = event.getSource();

        //添加中心
        String target=title;
        items.add(Item.builder().name(target).itemStyle(ItemStyle.builder().color("blue").build()).symbolSize("80").build());

        Class<?> clazz =event.getClass();
        Field[] fields = clazz.getDeclaredFields();
        String[] ignoredFields = {"title","type","id","updateTime","handleTime","createTime","content","level","address","source"};

        Set<String> addedItems = new HashSet<>();
        Set<String> addedLinks = new HashSet<>();

        for(Field field : fields){
            field.setAccessible(true);
            String fieldName = field.getName();

            if (isIgnored(fieldName, ignoredFields)) {
                continue;
            }

            try{
                String value = field.get(event).toString();
                String color = "#4d8ce5";

                if(fieldName.equals("state")){
                    if(value.equals("1")){
                        color = "red";
                        value = "未处理";
                    }
                    else {
                        color = "green";
                        value = "已处理";
                    }
                }

                if (!addedItems.contains(value)) {
                    items.add(Item.builder().name(value).itemStyle(ItemStyle.builder().color(color).build()).build());
                    addedItems.add(value);
                }

                String linkName = fieldName;
                if(fieldName.equals("alarmType")) continue;
                if(fieldName.equals("reportPerson")) linkName="上报人";
                if(fieldName.equals("handlePerson")) linkName="处理人";

                String linkKey = value + "_" + target + "_" + linkName;
                if (!addedLinks.contains(linkKey)) {
                    links.add(Link.builder().name(linkName).source(value).target(target).build());
                    addedLinks.add(linkKey);
                }

            }catch (Exception e){
                System.out.println("error");
            }
        }

        if(event.getHandlePerson()!=null){
            String linkKey = event.getHandlePerson() + "_" + event.getReportPerson() + "_" + "通知";
            if (!addedLinks.contains(linkKey)) {
                links.add(Link.builder().source(event.getHandlePerson()).target(event.getReportPerson()).name("通知").build());
                addedLinks.add(linkKey);
            }
        }

        //将地址加入并于事件连接，尺寸增大，设置颜色
        if (!addedItems.contains(address)) {
            items.add(Item.builder()
                    .name(address)
                    .symbolSize("60")
                    .itemStyle(ItemStyle.builder().color("#df8020").build())
                    .build());
            addedItems.add(address);
        }
        String linkKey = address + "_" + target + "_" + "地址";
        if (!addedLinks.contains(linkKey)) {
            links.add(Link.builder().name("地址").source(address).target(target).build());
            addedLinks.add(linkKey);
        }

        //连接 事件和事件来源设备
        //加入警报等级
        String alarmLevel = levelTransfor(event.getLevel());
        if (!addedItems.contains(alarmLevel)) {
            items.add(Item.builder().name(alarmLevel).itemStyle(ItemStyle.builder().color("#df209f").build()).build());
            addedItems.add(alarmLevel);
        }
        linkKey = source + "_" + alarmLevel + "_" + "警报等级";
        if (!addedLinks.contains(linkKey)) {
            links.add(Link.builder().name("警报等级").source(source).target(alarmLevel).build());
            addedLinks.add(linkKey);
        }
        linkKey = source + "_" + event.getAlarmType() + "_" + " ";
        if (!addedLinks.contains(linkKey)) {
            links.add(Link.builder().name(" ").source(source).target(event.getAlarmType()).build());
            addedLinks.add(linkKey);
        }
        linkKey = event.getAlarmType() + "_" + alarmLevel + "_" + " ";
        if (!addedLinks.contains(linkKey)) {
            links.add(Link.builder().name(" ").source(event.getAlarmType()).target(alarmLevel).build());
            addedLinks.add(linkKey);
        }

        return  ResultDTO.builder()
                .event(event)
                .links(links)
                .items(items)
                .build();
    }

    public ResultDTO getExtendRelation(ResultDTO resultDTO){
        String target = resultDTO.getEvent().getTitle();
        String address = resultDTO.getEvent().getAddress();

        //添加该地址下的下的多个检测器
        List<String> sensors = new ArrayList<>();
        sensors.addAll(eventRepository.findSpeedSensorsByAddress(address));
        sensors.addAll(eventRepository.findFlowSensorsByAddress(address));
        sensors.addAll(eventRepository.findCheckSensorsByAddress(address));
        sensors.addAll(eventRepository.findParkSensorsByAddress(address));
        sensors.addAll(eventRepository.findLaneSensorsByAddress(address));
        sensors.addAll(eventRepository.findEnvironmentSensorsByAddress(address));

        List<Link> sensorLinks = new ArrayList<>();
        List<Item> sensorItems = new ArrayList<>();

        Set<String> addedItems = new HashSet<>();
        Set<String> addedLinks = new HashSet<>();

        for(String sensor : sensors){
            if (!addedItems.contains(sensor)) {
                sensorItems.add(Item.builder()
                        .name(sensor)
                        .itemStyle(ItemStyle.builder().color(sensor.equals(resultDTO.getEvent().getSource()) ? "#dfbf20" : "#8b760e").build())
                        .symbolSize(sensor.equals(resultDTO.getEvent().getSource()) ? "50" : null)
                        .build());
                addedItems.add(sensor);
            }
            String linkKey = sensor + "_" + address + "_" + "监控";
            if (!addedLinks.contains(linkKey)) {
                sensorLinks.add(Link.builder().source(sensor).target(address).name("监控").build());
                addedLinks.add(linkKey);
            }
            if(sensor.equals(resultDTO.getEvent().getSource())){
                linkKey = sensor + "_" + target + "_" + "来源";
                if (!addedLinks.contains(linkKey)) {
                    sensorLinks.add(Link.builder().name("来源").source(sensor).target(target).build());
                    addedLinks.add(linkKey);
                }
            }
        }

        resultDTO.getLinks().addAll(sensorLinks);
        resultDTO.getItems().addAll(sensorItems);

        //上报人其他事件拓展
        String reportPerson = resultDTO.getEvent().getReportPerson();
        List<Event> events = eventRepository.findEventsByReportPerson(reportPerson,resultDTO.getEvent().getId());
        for(Event event : events){
            String eventTitle = event.getTitle();
            if (!addedItems.contains(eventTitle)) {
                resultDTO.getItems().add(Item.builder().name(eventTitle).itemStyle(ItemStyle.builder().color(event.getState()==1?"red":"green").build()).build());
                addedItems.add(eventTitle);
            }
            String linkKey = reportPerson + "_" + eventTitle + "_" + "上报";
            if (!addedLinks.contains(linkKey)) {
                resultDTO.getLinks().add(Link.builder().source(reportPerson).target(eventTitle).name("其他上报").build());
                addedLinks.add(linkKey);
            }
        }
        return resultDTO;
    }
}