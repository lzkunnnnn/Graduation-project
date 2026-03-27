import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;

import java.util.List;

public class jiebaExample {
    public static void main(String[] args) {
        JiebaSegmenter segmenter = new JiebaSegmenter();
        String text = "我来到北京清华大学";
        List<SegToken> tokens = segmenter.process(text, JiebaSegmenter.SegMode.INDEX);
        for (SegToken token : tokens) {
            System.out.println(token.word);
        }
    }
}