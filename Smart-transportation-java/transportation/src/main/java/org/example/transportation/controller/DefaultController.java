package org.example.transportation.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/")
public class DefaultController {
    @GetMapping("")
    String welcome(){
        return "welcome";
    }
}
