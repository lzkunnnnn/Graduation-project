package org.example.transportation.component;

import java.util.Random;

public class RandomCodeGenerator {
    // 生成指定长度的随机字符（字母或数字）
    public static String generateRandomSuffix(int length) {
        String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // 可选字符范围
        Random random = new Random();
        StringBuilder sb = new StringBuilder(length);

        for (int i = 0; i < length; i++) {
            int index = random.nextInt(characters.length()); // 随机选择一个字符
            sb.append(characters.charAt(index));
        }

        return sb.toString();
    }
}