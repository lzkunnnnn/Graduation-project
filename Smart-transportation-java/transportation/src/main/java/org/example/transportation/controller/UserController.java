package org.example.transportation.controller;


import org.example.transportation.dox.User;
import org.example.transportation.vo.ResultVO;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/")
public class UserController {
    @GetMapping("user")
    public ResultVO addUser(@RequestParam("name")String name) {
        return ResultVO.success(name);
    }
}
