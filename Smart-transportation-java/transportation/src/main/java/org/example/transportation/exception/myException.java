package org.example.transportation.exception;
import lombok.*;

@EqualsAndHashCode(callSuper = true)
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class myException extends RuntimeException{
    private String message;
    private int codeN;
    private Code code;//方便直接用封装码
}
