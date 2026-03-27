package org.example.transportation.dox.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EventHandleDTO {

    private int handledNum;
    private int unHandledNum;

    private int alarmNum;
    private int earlyAlarmNum;

    private int parkNum;
    private int checkNum;
    private int speedNum;
    private int flowNum;
    private int laneNum;
    private int environmentNum;

    private int handledLevelOne;
    private int handledLevelTwo;
    private int handledLevelThree;
    private int handledLevelFour;

    private int unHandledLevelOne;
    private int unHandledLevelTwo;
    private int unHandledLevelThree;
    private int unHandledLevelFour;
}
