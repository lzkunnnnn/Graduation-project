package org.example.transportation.dox.sensor;


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
public class SensorLane {
    @Id
    private int id;
    private String name;
    private String type;
    private int state;
    private int level;
    private String address;

    @ReadOnlyProperty
    private LocalDateTime updateTime;
    @ReadOnlyProperty
    private LocalDateTime createTime;
}
