import java.util.*;
public class Main {
    public static void main(String[] args) {
        Solution solution = new Solution();
        String s = "barfoofoobarthefoobarman";
        String[] words = {"bar","foo","the"};
        List<Integer> result = solution.findSubstring(s, words);
        System.out.println(result);
    }
}