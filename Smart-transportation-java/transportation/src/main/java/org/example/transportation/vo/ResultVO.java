package org.example.transportation.vo;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.example.transportation.exception.Code;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ResultVO {

    private int code;//业务码
    private String message; //异常信息，成功时不需要这个属性
    private Object data; //返回数据

    public static ResultVO success(Object data){
        return ResultVO.builder()
                .data(data)
                .code(200)
                .build();
    }

    public static ResultVO error(Code code){
        return ResultVO.builder()
                .code(code.getCode())
                .message(code.getMessage())
                .build();
    }
    public static ResultVO error(int code,String message){
        return ResultVO.builder()
                .code(code)
                .message(message)
                .build();
    }
    private static final ResultVO EMPTY = ResultVO.builder().code(200).build();
    public static ResultVO ok(){
        return EMPTY;
    }
}