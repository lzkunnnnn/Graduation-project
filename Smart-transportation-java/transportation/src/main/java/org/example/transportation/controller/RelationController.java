package org.example.transportation.controller;

import lombok.RequiredArgsConstructor;
import org.example.transportation.service.RelationService;
import org.example.transportation.vo.ResultVO;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/relation/")
@RequiredArgsConstructor
public class RelationController {

    private final RelationService relationService;


    @GetMapping("getRelation")
    public ResultVO getRelation(int id) {
        return ResultVO.success(relationService.getExtendRelation(relationService.getRelation(id)));
    }
}
