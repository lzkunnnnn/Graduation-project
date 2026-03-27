package org.example.transportation.exception;


import lombok.RequiredArgsConstructor;
import org.example.transportation.vo.ResultVO;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@RequiredArgsConstructor
public class ExceptionController {
    @ExceptionHandler(myException.class)
    public ResultVO myExcetion(myException e) {
        if(e.getCode()!=null){
            return ResultVO.error(e.getCode());
        }
        return ResultVO.error(e.getCodeN(),e.getMessage());
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResultVO HttpMessageNotReadableException(HttpMessageNotReadableException e) {
        System.out.println(e);
        return ResultVO.success(e);
    }


}
