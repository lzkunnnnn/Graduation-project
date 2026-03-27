package org.example.transportation.dox.relation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Link {
    private String source;
    private String target;
    private String name;
    private LineStyle lineStyle;
    private ItemStyle itemStyle;
}
