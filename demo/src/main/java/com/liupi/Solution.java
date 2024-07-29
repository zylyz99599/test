package com.liupi;

import java.util.HashMap;
import java.util.Map;

public class Solution {

    // region 报数公平性游戏
    /*
     * 现在有一个双人游戏，假设游戏的双方分别是 A 和 B，游戏规则如下：A 和 B 依次
     * 报数，报数的数字来自于 （这里假设 N 不超过 30）的集合，要求两个
     * 人报的数字序列不允许重复，现给定一个目标值 Target，如果最后人一个报数结束后，两
     * 个人报数得到的序列 满足：已报得数字得和大于Target ，那么最后一个报数的人就获胜。但是这个游戏存在如下情况，
     * 就是在给定 N 和 Target 情况下，A 和 B 都按照最优的
     * 方式报数，那么最先报数的人一定能获得胜利。请实现如下函数： bool isAlwaysWin(int N, int Target) ，
     * 判断给定 N 和 Target情况下，先报数的人是否一定取胜，True 表示一定取胜，False 表示不是。
     */
    public static boolean isAlwaysWin(int N, int Target) {
        Map<Integer, Boolean> memo = new HashMap<>();
        return canWin(N, Target, 0, 0, memo);
    }
    private static boolean canWin(int N, int Target, int total, int used, Map<Integer, Boolean> memo) {
        // 当前total以及超过了target，当前就已经输了
        if (total >= Target) {
            return false;
        }
        // 剪枝
        if (memo.containsKey(used)) {
            return memo.get(used);
        }
        for (int i = 1; i <= N; i++) {
            int currBit = 1 << i;
            if ((used & currBit) == 0) { // 判断i是否有被用过
                if (!canWin(N, Target, total + i, used | currBit, memo)) {
                    memo.put(used, true); // 如果对手输则自己必赢，记录memo
                    return true;
                }
            }
        }
        memo.put(used, false);
        return false;
    }


    // endregion

}
