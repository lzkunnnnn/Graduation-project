package org.example.transportation.dox.event;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.ReadOnlyProperty;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Event {

    @Id
    private int id;
    private String title;
    //sensor，系统类型
    private String type;
   //待处理状态
    private int state;
    //警报的类型
    private String alarmType;
    private int level;

    private String address;
    //内容
    private String content;
    //来源
    private String source;
    //上报人
    private String reportPerson;
    //处理人，负责人
    private String handlePerson;


    @ReadOnlyProperty
    private LocalDateTime updateTime;
    @ReadOnlyProperty
    private LocalDateTime createTime;
    @ReadOnlyProperty
    private LocalDateTime handleTime;
}
