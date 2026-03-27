package org.example.transportation.dox.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EventTypeDTO {
    private int smogNum;
    private int elecNum;
    private int gasNum;
    private int waterNum;
}
