package org.example.transportation.dox.relation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Item {
    private String name;
    private ItemStyle itemStyle;
    private String symbolSize;
}
