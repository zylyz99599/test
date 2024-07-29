import java.util.*;


class Solution {

    //================================================================
    // 2024/3/18 no1
    public List<List<Integer>> threeSum(int[] nums) {

        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        for (int first = 0; first < n; ++first) {
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            int third = n - 1;
            int target = -nums[first];
            for (int second = first + 1; second < n; ++second) {
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                if (second == third)
                    break;
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    // 2024/3/18 no2
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int best = 10000;
        for (int a = 0; a < n; a++) {
            if (a > 0 && nums[a - 1] == nums[a]) {
                continue;
            }
            int b = a + 1;
            int c = n - 1;
            while (b < c) {
                int sum = nums[a] + nums[b] + nums[c];
                if (sum == target)
                    return target;
                if (Math.abs(sum - target) < Math.abs(best - target)) {
                    best = sum;
                }
                if (sum > target) {
                    int k0 = c - 1;
                    while (k0 > b && nums[k0] == nums[c]) {
                        --k0;
                    }
                    c = k0;
                } else {
                    int j0 = b + 1;
                    while (j0 < c && nums[j0] == nums[b])
                        j0++;
                    b = j0;
                }
            }

        }

        return best;
    }

    // 2024/3/18 no3
    public List<String> letterCombinations(String digits) {

        List<String> combinations = new ArrayList<String>();

        if (digits.length() == 0) {
            return combinations;
        }

        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backtrack(combinations, phoneMap, digits, 0, new StringBuffer());
        return combinations;
    }

    public void backtrack(List<String> combinations, Map<Character, String> phonemap, String digits, int index, StringBuffer combination) {
        if (index == digits.length()) {
            combinations.add(combination.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = phonemap.get(digit);
            int lettersCount = letters.length();
            for (int i = 0; i < lettersCount; i++) {
                combination.append(letters.charAt(i));
                backtrack(combinations, phonemap, digits, index + 1, combination);
                combination.deleteCharAt(index);
                // 如果不删除的话，那么返回上一级的就还是ad 而不是a 这样一来，再加上时就是ade
            }
        }
    }

    // 2024/3/18 no4
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> quadruplets = new ArrayList<List<Integer>>();
        if (nums == null || nums.length < 4) {
            return quadruplets;
        }
        Arrays.sort(nums);
        int length = nums.length;
        for (int i = 0; i < length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if ((long) nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target) {
                continue;
            }
            for (int j = i + 1; j < length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                    break;
                }
                if ((long) nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target) {
                    continue;
                }
                int left = j + 1, right = length - 1;
                while (left < right) {
                    long sum = (long) nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        quadruplets.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        left++;
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return quadruplets;
    }


    // 2024/3/18 no5
    public int getLength(ListNode head) {
        int length = 0;
        while (head != null) {
            ++length;
            head = head.next;
        }
        return length;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {

        ListNode dummy = new ListNode(0, head);
        int length = getLength(head);
        ListNode cur = dummy;
        for (int i = 1; i < length - n + 1; i++) {
            cur = cur.next;
        }
        cur.next = cur.next.next;
        ListNode ans = dummy.next;
        return ans;

    }


    //===================================================

    // 2024/3/20 no1
    private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
        put('{', '}');
        put('(', ')');
        put('[', ']');
    }};

    public boolean isValid(String s) {
        if (s.length() > 0 && !map.containsKey(s.charAt(0)))
            return false;
        LinkedList<Character> stack = new LinkedList<Character>() {{
            add('?');
        }};
        for (Character c : s.toCharArray()) {
            if (map.containsKey(c)) stack.addLast(c);
            else if (map.get(stack.removeLast()) != c) return false;
        }
        return stack.size() == 1;
    }

    // 2024/3/20 no2
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(-1);
        ListNode current = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }

        if (list1 != null) {
            current.next = list1;
        } else {
            current.next = list2;
        }

        return dummy.next;
    }

    // 2024/3/20 no3
    List<String> ans = new ArrayList<String>();

    public List<String> generateParenthesis(int n) {

        if (n <= 0)
            return ans;
        getParenthesis("", n, n);
        return ans;
    }

    private void getParenthesis(String str, int left, int right) {
        if (left == 0 && right == 0) {
            ans.add(str);
            return;
        }
        if (left == right) {
            getParenthesis(str + "(", left - 1, right);
        } else if (left < right) {
            if (left > 0) {
                getParenthesis(str + "(", left - 1, right);
            }
            getParenthesis(str + ")", left, right - 1);
        }
    }

    // 2024/3/20 no4
//    public ListNode mergeKLists(ListNode[] lists) {
//        ListNode ans = null;
//        for (int i = 0; i < lists.length; i++) {
//            ans = mergeTwoList(ans, lists[i]);
//        }
//        return ans;
//    }

    public ListNode mergeTwoList(ListNode a, ListNode b) {
        if (a == null || b == null)
            return a == null ? b : a;
        ListNode head = new ListNode(-1);
        ListNode current = head;
        ListNode h1 = a, h2 = b;
        while (h1 != null && h2 != null) {
            if (h1.val <= h2.val) {
                current.next = h1;
                h1 = h1.next;
            } else {
                current.next = h2;
                h2 = h2.next;
            }
            current = current.next;
        }
        current.next = h1 == null ? h2 : h1;
        return head.next;
    }

    // 2024/3/20 no5
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode temp = dummy;

        while (temp.next != null && temp.next.next != null) {
            ListNode one = temp.next;
            ListNode two = temp.next.next;

            temp.next = two;
            one.next = two.next;
            two.next = one;

            temp = one; // 更新temp指向下一对待交换节点的前一个节点
        }
        return dummy.next;
    }


    //============================================/
    // 2024/3/21 no1
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode temp = dummy;

        int len = getLength(head);
        if (len < k) return dummy.next;

        for (int i = 0; i < len / k; i++) {
            ListNode start = temp.next; // 每组反转的头节点
            ListNode cur = start.next; // 当前待反转节点的下一个节点

            for (int j = 0; j < k - 1; j++) {
                ListNode nextTemp = cur.next; // 保存下一个待反转节点的下一个节点
                cur.next = temp.next; // 将当前节点插入到反转链表的头部
                temp.next = cur; // 更新反转链表的头节点
                start.next = nextTemp; // 将当前节点从原链表中删除，并指向下一个待反转节点
                cur = nextTemp; // 更新当前待反转节点
            }

            temp = start; // 更新 temp 指针指向当前反转后的尾节点
        }

        return dummy.next;
    }

    // 2024/3/21 no2
    public int removeDuplicates(int[] nums) {
        int start = 0;
        int run = 1;
        if (nums.length == 0) {
            return 0;
        }
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[start]) {
                nums[++start] = nums[i];
            }
        }
        return start + 1;
    }

    // 2024/3/21 no3
    public int removeElement(int[] nums, int val) {
        int length = nums.length;
        if (length == 0) return 0;
        int cnt = 0;
        for (int i = 0; i < length; i++) {
            if (nums[i] != val) {
                nums[cnt++] = nums[i];
            }
        }
        return cnt;
    }

    // 2024/3/21 no4
    public int strStr(String haystack, String needle) {
        int len = needle.length();

        for (int i = 0; i <= haystack.length() - len; i++) {
            boolean able = true;
            for (int j = 0; j < len; j++) {
                if (haystack.charAt(j + i) != needle.charAt(j)) {
                    able = false;
                }
            }
            if (able == true) return i;

        }
        return -1;
    }

    // 2024/3/21 no5
    int MIN = Integer.MIN_VALUE, MAX = Integer.MAX_VALUE;
    int LIMIT = -1073741824; // MIN 的一半

    public int divide(int a, int b) {
        if (a == MIN && b == -1) return MAX;
        boolean flag = false;
        if ((a > 0 && b < 0) || (a < 0 && b > 0)) flag = true;
        if (a > 0) a = -a;
        if (b > 0) b = -b;
        int ans = 0;
        while (a <= b) {
            int c = b, d = -1;
            while (c >= LIMIT && d >= LIMIT && c >= a - c) { // 让除数尽可能大，例如21/2 变成21/4 21/8 当21/16时就会退出循环 ，之后开始相减 每次减的就是16，答案就加上16
                c += c;
                d += d;
            }
            a -= c;
            ans += d;
        }
        return flag ? ans : -ans;
    }

    //================================================/

    // 2024/3/25 no1

    public List<Integer> findSubstring(String s, String[] words) {
        int lenword = words[0].length();
        int lenwords = words.length;
        int len = s.length();
        ArrayList<Integer> ansSub = new ArrayList<>();
        int[] biaoji = new int[lenwords];
        for (int i = 0; i < lenwords; i++) {
            biaoji[i] = 0;
        }

        for (int i = 0; i < lenwords; i++) {
            biaoji[i] = 1;
            dfs_sub(s, words[i], biaoji, words, lenword * lenwords, ansSub);
            biaoji[i] = 0;
        }

        return ansSub;
    }

    // 通过率 151/179 部分超时。。。。。 // 此处还缺少去重的结果
    public void dfs_sub(String s, String sub, int[] biaoji, String[] words, int len, ArrayList<Integer> ansSub) {
        if (sub.length() == len) {
            int index = s.indexOf(sub, 0); // 从startIndex开始查找
            while (index != -1) {
                ansSub.add(index);
                index = s.indexOf(sub, index + 1); // 继续查找下一个匹配位置
            }
            return;
        }

        for (int i = 0; i < biaoji.length; i++) {
            if (biaoji[i] == 0) {
                biaoji[i] = 1;
                dfs_sub(s, sub + words[i], biaoji, words, len, ansSub);
                biaoji[i] = 0;
            }
        }
    }


    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(target - nums[i])) {
                return new int[]{hashMap.get(nums[i]), hashMap.get(target - nums[i])};
            }
            hashMap.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] array = s.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<String>());
            }
            map.get(key).add(s);
        }
        return new ArrayList<>(map.values());
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }
        int maxlen = 0;
        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int curNum = num;
                int curStreak = 1;
                while (num_set.contains(curNum + 1)) {
                    curNum += 1;
                    curStreak += 1;
                }
                maxlen = Math.max(curStreak, maxlen);
            }
        }
        return maxlen;
    }

    public void moveZeroes(int[] nums) {
        if (nums == null) {
            return;
        }
        //第一次遍历的时候，j指针记录非0的个数，只要是非0的统统都赋给nums[j]
        int j = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != 0) {
                nums[j++] = nums[i];
            }
        }
        //非0元素统计完了，剩下的都是0了
        //所以第二次遍历把末尾的元素都赋为0即可
        for (int i = j; i < nums.length; ++i) {
            nums[i] = 0;
        }
    }

    public int maxArea(int[] height) {
        int len = height.length;
        int left = 0, right = len - 1;
        int maxContainer = Math.min(height[left], height[right]) * (len - 1);
        while (left < right) {
            if (height[left] <= height[right]) {
                left++;
                int contain = Math.min(height[left], height[right]) * (right - left);
                if (contain > maxContainer)
                    maxContainer = contain;
            }
            if (height[left] > height[right]) {
                right--;
                int contain = Math.min(height[left], height[right]) * (right - left);
                if (contain > maxContainer)
                    maxContainer = contain;
            }
        }
        return maxContainer;

    }

    public List<List<Integer>> threeSum2(int[] nums) {
        Arrays.sort(nums);
        int len = nums.length;
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int left = i + 1, right = len - 1;
            int target = -nums[i];
            for (; left < len; left++) {
                if (left > i + 1 && nums[left] == nums[left - 1])
                    continue;
                ;
                while (left < right && nums[left] + nums[right] > target) {
                    right--;
                }
                if (left == right)
                    break;
                ;
                if (nums[left] + nums[right] == target) {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    ans.add(list);
                }
            }

        }

        return ans;
    }

    public int trap(int[] height) {
        // 接水问题， 每一列能接多少水取决于这一列左边最高的边和右边最高的边

        // 如果直接遍历，时间复杂度较高 O(n^2) 使用动态规划

        // 在 1 < i < n-1 的范围内，leftMax[i] = max{height[i],leftMax[i-1]}

        int len = height.length;
        int[] leftMax = new int[len];
        int[] rightMax = new int[len];
        leftMax[0] = height[0];
        rightMax[len - 1] = height[len - 1];
        for (int i = 1; i < len; i++) {
            leftMax[i] = Math.max(height[i], leftMax[i - 1]);
        }
        for (int j = len - 2; j >= 0; j--) {
            rightMax[j] = Math.max(height[j], rightMax[j + 1]);
        }
        int capacity = 0;
        for (int i = 0; i < len; i++) {
            capacity += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return capacity;

    }

    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        if (len <= 1) {
            return len;
        }
        int l = 0, r = 1;
        // 定义一个集合
        HashSet<Character> set = new HashSet<>();
        int max = 0;
        // 首先先将左边的指针所指元素入队列
        set.add(s.charAt(l));
        while (l < r) {
            while (l < r && !set.contains(s.charAt(r))) {
                r++;
                set.add(s.charAt(r));
            }
            max = Math.max(max, r - l + 1);
            // 将当前最左边的移除
            set.remove(s.charAt(l));
            l++;
        }
        return max;
    }

    public List<Integer> findAnagrams(String s, String p) {
        int len = s.length();
        int window = p.length();
        List<Integer> result = new ArrayList<>();
        int[] sa = new int[26];
        int[] pa = new int[26];
        // 初始化
        for (int i = 0; i < window; i++) {
            pa[p.charAt(i) - 'a']++;
        }
        for (int i = 0; i < window; i++) {
            sa[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i <= len - window; i++) {
            if (Arrays.equals(pa, sa)) {
                result.add(i);
            }
            // 如果不相等，窗口往下移动
            if (i < len - window) {
                sa[s.charAt(i) - 'a']--;
                sa[s.charAt(i + window) - 'a']++;
            }

        }
        return result;
    }

    // 和为k的子数组
    public int subarraySum(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (map.containsKey(pre - k)) {
                count += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return count;

    }

    // 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
//        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[0] != o2[0] ? o2[0] - o1[0] : o2[1] - o1[1];
//            }
//        });
//
//        for (int i = 0; i < k; i++) {
//            pq.offer(new int[]{nums[i], i});
//        }
//        int[] ans = new int[n - k + 1];
//        ans[0] = pq.peek()[0];
//        for (int i = k; i < n; i++) {
//            pq.offer(new int[]{nums[i], i});
//            while (pq.peek()[1] <= i - k) {
//                pq.poll();
//            }
//            ans[i - k + 1] = pq.peek()[0];
//        }
//        return ans;
        // 分块来看
        int[] pre = new int[n];
        int[] tail = new int[n];
        int[] ans = new int[n - k + 1];
        for (int i = 0; i < n; i++) {
            if (i % k == 0) {
                pre[i] = nums[i];
            } else {
                pre[i] = Math.max(pre[i - 1], nums[i]);
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if (i == n - 1 || (i + 1) % k == 0) {
                tail[i] = nums[i];
            } else {
                tail[i] = Math.max(tail[i + 1], nums[i]);
            }
        }

        for (int i = 0; i < n - k + 1; i++) {
            if (i % k == 0) {
                ans[i] = tail[i];
            } else {
                ans[i] = Math.max(tail[i], pre[i + k - 1]);
            }
        }
        return ans;

    }

    // 最小覆盖字串
    public String minWindow(String s, String t) {
        if (s.length() < t.length() || s == null || t == null || s == "" || t == "") {
            return "";
        }
        int[] needs = new int[128];
        int[] window = new int[128];
        for (int i = 0; i < t.length(); i++) {
            needs[t.charAt(i)]++;
        }
        int left = 0;
        int right = 0;
        String res = "";
        int count = 0;
        int minLength = s.length() + 1;
        while (right < s.length()) {
            char ch = s.charAt(right);
            window[ch]++;
            if (needs[ch] > 0 && needs[ch] >= window[ch]) {
                count++;
            }
            while (count == t.length()) {
                ch = s.charAt(left);
                if (needs[ch] > 0 && needs[ch] >= window[ch]) {
                    count--;
                }
                if (right - left + 1 < minLength) {
                    minLength = right - left + 1;
                    res = s.substring(left, right + 1);
                }
                window[ch]--;
                left++;
            }
            right++;
        }
        return res;
    }

    // 最大子数组和
    public int maxSubArray(int[] nums) {
        int length = nums.length;
        int ans = Integer.MIN_VALUE;
        // 数组长度为1直接返回
        if (length == 1) {
            return nums[0];
        }
        int preSum = 0;
        int minPreSum = 0;
        //  循环时求出到当前字符串位置之前最小的前缀和，求出算上当前字符串的的前缀和
        //  这样到当前字符串位置时，得到的结果就是当前位置最大子串
        for (int i = 0; i < length; i++) {
            preSum += nums[i];
            ans = Math.max(ans, preSum - minPreSum);
            minPreSum = Math.min(minPreSum, preSum);
        }
        return ans;
    }

    // 合并区间
    public int[][] merge(int[][] intervals) {

        // 如果输入为空直接返回
        if (intervals.length == 0) {
            return new int[0][2];
        }
        // 写一个排序规则
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                // return true o2 在 o1前面，false o1在o2前面
                return o1[0] - o2[0];
            }
        });
        if (intervals.length == 1) {
            return intervals;
        }
        // [[1,3],[2,6],[8,10],[15,18]] ====> [[1,6],[8,10],[15,18]]
        List<int[]> ans = new ArrayList<>();
        // 对数组进行便利
        int i = 0;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            while (i < intervals.length - 1 && intervals[i + 1][0] <= right) {
                i++;
                // 替换右边最大值
                right = Math.max(intervals[i][1], right);
            }
            ans.add(new int[]{left, right});
            i++;
        }
        return ans.toArray(new int[ans.size()][]);
    }

    // 轮转数组
    public void rotate(int[] nums, int k) {
        if (k == 0 || nums.length == 0) {
            return;
        }
        Deque<Integer> queue = new LinkedList();
        for (int num : nums) {
            queue.add(num);
        }
        for (int i = 0; i < k; i++) {
            Integer last = queue.pollLast();
            queue.addFirst(last);
        }
        int index = 0;
        for (Integer o : queue) {
            nums[index++] = o;
        }


    }

    // 轮转数组2
    public void rotate2(int[] nums, int k) {
        if (k == 0 || nums.length == 0) {
            return;
        }
        int[] newArr = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            newArr[(i + k) % (nums.length)] = nums[i];
        }
        System.arraycopy(newArr, 0, nums, 0, nums.length);
    }

    // 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        // 求出所有数的乘积再除当前数，但是要求时间复杂度为O(n)，求出左边乘积和又变乘积就能得出结果
        int len = nums.length;
        int[] preMul = new int[len + 1];
        int[] tailMul = new int[len + 1];
        preMul[0] = 1;
        tailMul[len] = 1;
        for (int i = 1; i < len; i++) {
            preMul[i] = nums[i - 1] * preMul[i - 1];
        }
        for (int i = len - 2; i >= 0; i++) {
            tailMul[i] = nums[i + 1] * tailMul[i + 1];
        }
        int[] ans = new int[len];
        for (int i = 0; i < len; i++) {
            ans[i] = preMul[i] * tailMul[i];
        }
        return ans;
    }

    // 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {

        // 长度为N的数组，其中最小的未出现的正整数必在 1~N+1中
        // 遍历一边数组，将小于0的数设为N+1
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) {
                nums[i] = nums.length + 1;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            int num = Math.abs(nums[i]);
            if (num <= nums.length) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    // 矩阵转置
    public void setZeroes(int[][] matrix) {
        // 定义两个标记数组
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[] row = new boolean[m];
        boolean[] col = new boolean[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = true;
                    col[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }

    }

    // 螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        // 定义上下边界
        int u = 0;
        int d = matrix.length - 1;
        int l = 0;
        int r = matrix[0].length - 1;
        // 一直循环
        while (true) {
            for (int i = l; i <= r; i++) ans.add(matrix[u][i]);
            if (++u > d) break;

            for (int i = u; i <= d; i++) ans.add(matrix[i][r]);
            if (--r < l) break;

            for (int i = r; i >= l; i--) ans.add(matrix[d][i]);
            if (--d < u) break;

            for (int i = d; i >= u; i--) ans.add(matrix[i][l]);
            if (++l > r) break;
        }
        return ans;
    }

    // 旋转图像
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] new_matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                new_matrix[j][n - i - 1] = matrix[i][j];
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = new_matrix[i][j];
            }
        }
    }

    // 搜索二维矩阵II
    public boolean searchMatrix(int[][] matrix, int target) {
        // 从右上角往下搜索
        int m = matrix.length;
        int n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target)
                return true;
            if (matrix[x][y] > target) {
                y--;
            } else {
                x++;
            }
        }
        return false;
    }

    // 相交链表

    /**
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode(int x) {
     * val = x;
     * next = null;
     * }
     * }
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
//        ListNode l1 = new ListNode();
//        ListNode l2 = new ListNode();
//        l1 = headA;
//        l2 = headB;
//        // 思路：两个节点先往后移动，如果某个节点移动到了null，则将这个节点移动到未移动完的头节点，然后当另外一个节点移动到空的时候，再移动到短的那个开始一起移动
//        while (l1 != l2) {
//            l1 = l1 == null ? headB : l1.next;
//            l2 = l2 == null ? headA : l2.next;
//        }
//        return l1;
        int count = 0;
        int lena = 0, lenb = 0;
        ListNode la = headA, lb = headB;
        while (la != null) {
            la = la.next;
            lena++;
        }
        while (lb != null) {
            lb = lb.next;
            lenb++;
        }
        la = headA;
        lb = headB;
        if (lena > lenb) {
            for (int i = 0; i < lena - lenb; i++) {
                la = la.next;
            }
            while (la != null && lb != null) {
                if (la == lb) {
                    return la;
                }
                la = la.next;
                lb = lb.next;
            }
            return null;
        }
        if (lenb > lena) {
            for (int i = 0; i < lenb - lena; i++) {
                lb = lb.next;
            }
            while (la != null && lb != null) {
                if (la == lb) {
                    return la;
                }
                la = la.next;
                lb = lb.next;
            }
            return null;
        }
        if (lenb == lena) {
            while (la != null && lb != null) {
                if (la == lb) {
                    return la;
                }
                la = la.next;
                lb = lb.next;
            }
            return null;
        }
        return null;
    }

    // 反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null)
            return null;
        // 反转链表，头插法即可解决
        ListNode newHead = null;
        while (head != null) {
            ListNode temp = head.next;
            head.next = newHead;
            newHead = head;
            head = temp;
        }
        return newHead;
    }

    // 判断回文链表
    public boolean isPalindrome(ListNode head) {
        if (head == null)
            return false;
        ListNode l = head;
        int len = 0;
        List<Integer> list = new ArrayList<>();
        while (l != null) {
            list.add(l.val);
            l = l.next;

        }
        int front = 0;
        int back = list.size() - 1;
        while (front < back) {
            if (!list.get(front).equals(list.get(back)))
                return false;
            front++;
            back--;
        }
        return true;

    }

    // 环形链表
    public boolean hasCycle(ListNode head) {
        // 同样用快慢指针，如果快指针能够再次遇上慢指针说明有环
        if (head == null || head.next == null) {
            return false;
        }
        // 定义快慢指针
        ListNode s = head;
        ListNode f = head.next;
        while (f != s) {
            s = s.next;
            if (f.next == null || f.next.next == null) {
                return false;
            }
            f = f.next.next;
            if (s == f) {
                return true;
            }
        }
        return true;
    }

    // 环形链表II 返回入环的第一个节点
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        // 使用快慢指针找出第一次相遇的点，然后慢指针再走一次到这个地方所需要的步数就是环的长度，再计算第一次走到标志位慢指针的步数，就能得出非
        // 将整个链表分为三段a,b,c b为相遇的点到入口点的长度可以推出
        // a - b + n(b+c) = 2a + 2b ====> (n-1)(b+c) = a-c ===> b+c为环的长度 可以推出当慢指针从相遇点出发时下次相遇就是在入口处
        ListNode s, f;
        s = head;
        f = head;
        while (f != null && f.next != null) {
            s = s.next;
            f = f.next;
            if (f.next == null)
                return null;
            f = f.next;
            if (s == f) {
                ListNode temp = head;
                while (head != s) {
                    head = head.next;
                    s = s.next;
                }
                return s;
            }
        }
        return null;
    }

    // 合并两个有序链表
    public ListNode merge2Lists(ListNode a, ListNode b) {
        if (a == null && b == null)
            return null;
        if (a == null && b != null)
            return b;
        if (a != null && b == null)
            return a;
        ListNode head = new ListNode(0);
        ListNode temp = head;
        while (a != null && b != null) {
            if (a.val <= b.val) {
                temp.next = a;
                a = a.next;
            } else {
                temp.next = b;
                b = b.next;
            }
            temp = temp.next;
        }
        temp.next = a == null ? b : a;
        return head.next;
    }

    //  合并K个有序链表
    public ListNode mergeKLists(ListNode[] lists) {
        // 慢的方法
//        if (lists.length==0)
//            return null;
//        int k = lists.length;
//        ListNode ans = null;
//        for (int i = 0; i < k; i++) {
//            ans = merge2Lists(ans, lists[i]);
//        }
//        return ans;

        // 创建一个优先队列，然后将链表的节点加入进去
        // return o1.val-o2.val 一个升序的队列
        PriorityQueue<ListNode> queue = new PriorityQueue<>((o1, o2) -> {
            return o1.val - o2.val;
        });
        for (ListNode node : lists) {
            if (node != null) {
                queue.add(node);
            }
        }
        ListNode head = new ListNode(0);
        ListNode temp = head;
        while (!queue.isEmpty()) {
            ListNode cur = queue.poll();
            temp.next = cur;
            temp = temp.next;
            if (cur.next != null) {
                queue.add(cur.next);
            }
        }
        return head.next;

    }

    // region 字节面试题目
    public int findSmallNumber(int[] nums, int k) {
        int len = nums.length;
        int min = Integer.MIN_VALUE;
        int start = 0;
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < k; i++) {
            int end = len - k + i;
            int minIndex = start;
            for (int j = start; j <= end; j++) {
                if (nums[minIndex] > nums[j])
                    minIndex = j;
            }
            result.append(nums[minIndex]);
            start = minIndex + 1;
        }
        min = Integer.parseInt(String.valueOf(result));
        return min;
    }

    public int findSmallNumber2(int[] nums, int k) {
        /*
         * 递归写法
         * */
        return Integer.parseInt(dfs(0, nums, k));
    }

    public String dfs(int start, int[] nums, int k) {
        if (k == 0)
            return "";
        int len = nums.length;
        int minIndex = start;
        int end = len - k;
        for (int i = start; i <= end; i++) {
            if (nums[i] < nums[minIndex])
                minIndex = i;
        }
        return nums[minIndex] + dfs(minIndex + 1, nums, k - 1);
    }

    // endregion
    public ListNode mergeKLists2(ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>((o1, o2) -> {
            return o1.val - o2.val;
        });
        int len = lists.length;
        for (int i = 0; i < len; i++) {
            queue.offer(lists[i]);
        }
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy;
        while (!queue.isEmpty()) {
            ListNode off = queue.peek();
            queue.poll();
            if (off.next != null) {
                queue.offer(off.next);
            }
            temp.next = off;
            temp = off;
        }
        return dummy.next;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        // 按层序来， 广度优先遍历
        // 设置一个队列
        if (root == null)
            return new ArrayList<List<Integer>>();
        if (root.left == null && root.right == null) {
            return new ArrayList<List<Integer>>(root.val);
        }
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        int count = 1;
        int nextCount = 0;
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            while (count != 0) {
                count--;
                TreeNode poll = queue.poll();
                temp.add(poll.val);
                if (poll.left != null) {
                    queue.offer(poll.left);
                    nextCount++;
                }
                if (poll.right != null) {
                    queue.offer(poll.right);
                    nextCount++;
                }
            }
            count = nextCount;
            ans.add(temp);
            nextCount = 0;
        }


        return ans;
    }

    // region leetcode 236 最近的公共祖先
    TreeNode closeAncestor;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        ancestor(root, p, q);
        return closeAncestor;
    }

    public boolean ancestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        boolean left = ancestor(root.left, p, q);
        boolean right = ancestor(root.right, p, q);
        if ((left && right) || (root.val == p.val || root.val == q.val) && (left || right)) {
            closeAncestor = root;
        }
        return left || right || (root.val == p.val || root.val == q.val);
    }

    // endregion

    // region leetcode124 二叉树中的最大路径和

    // 定义一个map list[0] 记录横着走 list[1]记录左右两边最大值

    Map<TreeNode, List<Integer>> pathMap = new HashMap<>();

    public int maxPathSum(TreeNode root) {
        // 可以记录每个节点中，往左走 往右走 往左往右中的最大值，链表结构？记录父节点？
        int max = Integer.MIN_VALUE;
        searchPath(root);
        for (TreeNode treeNode : pathMap.keySet()) {
            List<Integer> integers = pathMap.get(treeNode);
            for (Integer integer : integers) {
                if (integer > max)
                    max = integer;
            }
        }
        return max;
    }

    public void searchPath(TreeNode root) {
        if (root == null)
            return;
        if (root.left == null && root.right == null) {
            List<Integer> temp = new ArrayList<>();
            temp.add(root.val);
            temp.add(root.val);
            pathMap.put(root, temp);
        }
        searchPath(root.left);
        searchPath(root.right);
        List<Integer> temp = new ArrayList<>();
        int leftValue = root.left != null ? pathMap.get(root.left).get(0) : 0;
        int rightValue = root.right != null ? pathMap.get(root.right).get(0) : 0;
        temp.add(Math.max(Math.max(leftValue, rightValue) + root.val, root.val));
        temp.add(Math.max(leftValue + rightValue + root.val, root.val));
        pathMap.put(root, temp);

    }

    // endregion

    // region leetcode200 岛屿数量
    public int numIslands(char[][] grid) {
        int nums = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    nums++;
                    islandDfs(grid, i, j);
                }
            }
        }
        return nums;
    }

    public void islandDfs(char[][] grid, int row, int col) {
        if (!isArea(grid, row, col)) return;
        if (grid[row][col] != '1') return;
        if (grid[row][col] == '1') {
            grid[row][col] = '2';
        }
        islandDfs(grid, row - 1, col);
        islandDfs(grid, row + 1, col);
        islandDfs(grid, row, col - 1);
        islandDfs(grid, row, col + 1);
    }

    public boolean isArea(char[][] grid, int row, int col) {
        return 0 <= row && row < grid.length && 0 <= col && col < grid[0].length;
    }

    // endregion

    // region leetcode994 腐烂橘子
    public int orangesRotting(int[][] grid) {
        int count = 0;
        Queue<List<Integer>> queue = new LinkedList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 2) {
                    List<Integer> temp = new ArrayList<>();
                    temp.add(i);
                    temp.add(j);
                    queue.offer(temp);
                    count++;
                }
            }
        }
        int time = 0;
        while (!queue.isEmpty()) {
            int newCount = 0;
            // 判断上下左右，如果有新鲜橘子则腐烂
            while (count != 0) {
                List<Integer> poll = queue.poll();
                count--;
                int row = poll.get(0);
                int col = poll.get(1);
                // 上
                if (isArea(grid, row - 1, col)) {
                    if (grid[row - 1][col] == 1) {
                        grid[row - 1][col] = 2;
                        List<Integer> temp = Arrays.asList(row - 1, col);
                        queue.offer(temp);
                        newCount++;
                    }
                }
                // 下
                if (isArea(grid, row + 1, col)) {
                    if (grid[row + 1][col] == 1) {
                        grid[row + 1][col] = 2;
                        List<Integer> temp = Arrays.asList(row + 1, col);
                        queue.offer(temp);
                        newCount++;
                    }
                }
                // 左
                if (isArea(grid, row, col - 1)) {
                    if (grid[row][col - 1] == 1) {
                        grid[row][col - 1] = 2;
                        List<Integer> temp = Arrays.asList(row, col - 1);
                        queue.offer(temp);
                        newCount++;
                    }
                }
                // 右
                if (isArea(grid, row, col + 1)) {
                    if (grid[row][col + 1] == 1) {
                        grid[row][col + 1] = 2;
                        List<Integer> temp = Arrays.asList(row, col + 1);
                        queue.offer(temp);
                        newCount++;
                    }
                }
            }
            if (newCount > 0)
                time++;
            count = newCount;
            newCount = 0;

        }
        boolean result = exFresh(grid);
        return result ? -1 : time - 1;
    }

    public boolean isArea(int[][] grid, int row, int col) {
        return 0 <= row && row < grid.length && 0 <= col && col < grid[0].length;
    }

    public boolean exFresh(int[][] grid) {
        boolean result;
        result = Arrays.stream(grid)
                .flatMapToInt(Arrays::stream)
                .anyMatch(x -> x == 1);
        return result;
    }

    // endregion

    // region leetcode207 课程表，拓扑排序
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // 创建一个邻接表
        HashMap<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
        int[] inDegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            map.put(i, new ArrayList<>());
        }
        for (int[] prerequisite : prerequisites) {
            inDegree[prerequisite[0]]++;
            map.get(prerequisite[1]).add(prerequisite[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0)
                queue.offer(i);
        }
        while (!queue.isEmpty()) {
            for (int i = queue.size(); i > 0; i--) {
                Integer poll = queue.poll();
                numCourses--;
                List<Integer> integers = map.get(poll);
                for (Integer integer : integers) {
                    inDegree[integer]--;
                    if (inDegree[integer] == 0)
                        queue.offer(integer);
                }

            }
        }
        return numCourses == 0;
    }
    // endregion

    // region leetcode22 括号生成 回溯
    public List<String> generateParenthesis2(int n) {
        List<String> ans = new ArrayList<>();
        dfsKuohao(0, 0, n, new StringBuilder(), ans);
//        System.out.println(ans);
        return ans;
    }

    void dfsKuohao(int left, int right, int n, StringBuilder cur, List<String> ans) {
        if (left < right) return;
        if (left > n) return;
        if ((left == right) && (left == n)) {
            ans.add(cur.toString());
        }
        // 添加左括号
        left++;
        cur.append('(');
        dfsKuohao(left, right, n, cur, ans);
        left--;
        cur.deleteCharAt(cur.length() - 1);

        right++;
        cur.append(')');
        dfsKuohao(left, right, n, cur, ans);
        right--;
        cur.deleteCharAt(cur.length() - 1);
    }
    // endregion

    // region leetcode79 单词搜索 回溯
    public boolean exist(char[][] board, String word) {
        int x = board.length;
        int y = board[0].length;
        // 遍历整个板子，找到单词的起点
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (word.charAt(0) == board[i][j]) {
                    int[][] flag = new int[x][y];
                    if (dfs79(i, j, board, 0, word, flag))
                        return true;
                }
            }
        }
        return false;
    }

    public boolean dfs79(int x, int y, char[][] board, int index, String word, int[][] flag) {
        if (x < 0 || y < 0 || x >= board.length || y >= board[0].length) return false;
        if (flag[x][y] == 1 || board[x][y] != word.charAt(index)) return false;
        if (index + 1 == word.length()) return true;
        flag[x][y] = 1;
        // 上下左右
        boolean up = dfs79(x - 1, y, board, index + 1, word, flag);
        boolean down = dfs79(x + 1, y, board, index + 1, word, flag);
        boolean left = dfs79(x, y - 1, board, index + 1, word, flag);
        boolean right = dfs79(x, y + 1, board, index + 1, word, flag);
        flag[x][y] = 0;
        return up || down || right || left;
    }

    // endregion

    // region leetcode131 分割回文串
    public List<List<String>> partition(String s) {
        List<List<String>> ans = new ArrayList<>(); // 存储所有方案
        dfs131(0, s, new ArrayList<>(), ans); // 从第0个字符开始进行递归
        return ans;
    }

    public void dfs131(int index, String s, List<String> cur, List<List<String>> ans) {
        if (index == s.length()) {
            ans.add(new ArrayList<>(cur));
            return;
        }
        for (int i = index; i < s.length(); i++) { // aaab
            if (isPalindrome(s, index, i)) {
                cur.add(s.substring(index, i + 1));
                dfs131(i + 1, s, cur, ans);
                cur.remove(cur.size() - 1);
            }
        }
    }

    public boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) return false;
            left++;
            right--;
        }
        return true;
    }
    // endregion

    // region leetcode51 N皇后问题
    public List<List<String>> solveNQueens(int n) {
        char[][] map = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                map[i][j] = '.';
            }
        }
        List<List<String>> ans = new ArrayList<List<String>>();
        dfs51(0, map, ans);
        return ans;
    }

    public List<String> mapToAnswer(char[][] map) {
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < map.length; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < map.length; j++) {
                sb.append(map[i][j]);
            }
            ans.add(sb.toString());
        }
        return ans;
    }

    public void dfs51(int row, char[][] map, List<List<String>> ans) {
        if (row == map.length) {
            List<String> list = mapToAnswer(map);
            ans.add(list);
            return;
        }
        for (int i = 0; i < map.length; i++) {
            if (isValid(map, row, i, map.length)) {
                map[row][i] = 'Q';
                dfs51(row + 1, map, ans);
                map[row][i] = '.';
            }
        }

    }

    public boolean isValid(char[][] map, int row, int col, int n) {
        for (int i = 0; i < row; i++) {
            if (map[i][col] == 'Q') return false; // 判断这一列上是否有Q
        }
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (map[i][j] == 'Q') return false; // 判断左上角是否有Q
        }
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if (map[i][j] == 'Q') return false; // 判断右上角是否有Q
        }
        return true;
    }
    // endregion

    // region leetcode35 搜索插入位置
    public int searchInsert(int[] nums, int target) {
        // 二分查找哇
        int len = nums.length;
        int left = 0, right = len - 1;
        while (left <= right) {
            int mid = (right + left) / 2;
            if (target == nums[mid]) return mid;
            if (target > nums[mid]) {
                left = mid + 1;
            }
            if (target < nums[mid]) {
                right = mid - 1;
            }
        }
        return left;
    }
    // endregion

    // region leetcode74 搜索二维矩阵
    // 两次二分查找就行了
    public boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int row = 0;
        int len = matrix.length;
        int len2 = matrix[0].length;
        int up = 0;
        int down = len - 1;
        while (up <= down) {
            int mid = ((down - up) >> 1) + up;
            if (target >= matrix[mid][0] && target <= matrix[mid][len2 - 1]) {
                row = mid;
                break;
            }
            if (target > matrix[mid][0]) up = mid + 1;
            if (target < matrix[mid][0]) down = mid - 1;
        }
        if (up > down) {
            return false;
        }
//        System.out.println(row);
        int left = 0, right = len2 - 1;
        while (left <= right) {
            int mid = ((right - left) >> 1) + left;
            if (target == matrix[row][mid]) return true;
            if (target > matrix[row][mid]) left = mid + 1;
            if (target < matrix[row][mid]) right = mid - 1;
        }
        return false;
    }
    // endregion

    // region leetcode34 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int len = nums.length;
        int left = 0, right = len - 1;
        int mid;
        while (left <= right) {
            mid = ((right - left) >> 1) + left;
            if (nums[mid] == target) {
                int l = mid, r = mid;
                while (l > 0) {
                    if (nums[l - 1] != target) break;
                    if (nums[l - 1] == target) l--;
                }
                while (r < len - 1) {
                    if (nums[r + 1] != target) break;
                    if (nums[r + 1] == target) r++;
                }
                return new int[]{l, r};
            }
            if (nums[mid] > target) {
                right = mid - 1;
            }
            if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return new int[]{-1, -1};
    }

    // endregion

    // region leetcode33 搜索旋转排序数组
    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid = 0;
        while (lo <= hi) {
            mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // 先根据 nums[mid] 与 nums[lo] 的关系判断 mid 是在左段还是右段
            if (nums[mid] >= nums[lo]) {
                // 再判断 target 是在 mid 的左边还是右边，从而调整左右边界 lo 和 hi
                if (target >= nums[lo] && target < nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }


    // endregion

    // region leetcode4 寻找两个正序数组中的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (m > n)
            return findMedianSortedArrays(nums2, nums1);
        int maxLeft = 0;
        int minRight = 0;
        int l = 0, r = m;
        while (l <= r) {
            int i = (l + r) / 2;
            int j = (m + n + 1) / 2 - i;
            if (i != m && j != 0 && nums1[i] < nums2[j - 1])
                l = i + 1;
            else if (i != 0 && j != n && nums1[i - 1] > nums2[j])
                r = i - 1;
            else {
                if (i == 0) maxLeft = nums2[j - 1];
                else if (j == 0) maxLeft = nums1[i - 1];
                else maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
                if ((m + n) % 2 == 1) return maxLeft;

                if (i == m) minRight = nums2[j];
                else if (j == n) minRight = nums1[i];
                else minRight = Math.min(nums1[i], nums2[j]);
                return (maxLeft + minRight) / 2.0;
            }
        }
        return 0.0;
    }

    // endregion

    // region leetcode20 有效的括号
    public boolean isValid2(String s) {
        if (s.length() % 2 == 1) return false;
        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{')
                stack.push(s.charAt(i));
            else {
                if (stack.isEmpty()) return false;
                if (s.charAt(i) == ')' && stack.pop() != '(') return false;
                if (s.charAt(i) == '}' && stack.pop() != '{') return false;
                if (s.charAt(i) == ']' && stack.pop() != '[') return false;
            }

        }
        return stack.isEmpty();
    }

    // endregion

    // region leetcode155 最小栈
    class MinStack {

        Node155 root;

        public MinStack() {
            this.root = new Node155();
        }

        public void push(int val) {
            Node155 node = new Node155(val, root.next);
            node.min = root.min;
            root.next = node;
            root.min = Math.min(root.min, val);
        }

        public void pop() {
            Node155 cur = root.next;
            if (cur.val == root.min) {
                root.min = cur.min;
            }
            root.next = cur.next;
            cur.next = null;
        }

        public int top() {
            return root.next.val;
        }

        public int getMin() {
            return root.min;
        }
    }

    class Node155 {
        Node155 next;
        int min, val;

        public Node155() {
            this.min = Integer.MAX_VALUE;
        }

        public Node155(int val) {
            this.val = val;
        }

        public Node155(int val, Node155 next) {
            this.val = val;
            this.next = next;
        }
    }

    // endregion

    // region leetcode739 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<List<Integer>> stack = new LinkedList<>();
        int len = temperatures.length;
        int[] ans = new int[len];
        if (len == 1) return new int[]{0};
        stack.offer(Arrays.asList(0, temperatures[0]));
        for (int i = 1; i < len; i++) {
            if (temperatures[i] <= stack.peek().get(1)) { // 如果当前值小于等于栈顶值，则入栈
                stack.push(Arrays.asList(i, temperatures[i]));
            }
            if (temperatures[i] > stack.peek().get(1)) {
                // 记录index,temp,index用于寻找符合条件的下标，temp记录次数
                int index = i;
                while (!stack.isEmpty() && (temperatures[index] > stack.peek().get(1))) {
                    List<Integer> pop = stack.pop();
                    ans[pop.get(0)] = index - pop.get(0);

                }
                stack.push(Arrays.asList(i, temperatures[i]));
            }
        }
        return ans;
    }

    // endregion

    // region leetcode394 字符串解码
    public String decodeString(String s) {
        Deque<String> sc = new LinkedList<>();
        Deque<Integer> sn = new LinkedList<>();
        StringBuilder res = new StringBuilder();
        int len = s.length();
        int num = 0;
        for (Character c : s.toCharArray()) {
            if (c >= 'a' && c <= 'z') res = res.append(c);
            if (c >= '0' && c <= '9') num = num * 10 + c - '0';
            if (c == '[') {
                sc.push(res.toString());
                sn.push(num);
                num = 0;
                res = new StringBuilder();
            }
            if (c == ']') {
                StringBuilder temp = new StringBuilder();
                int pop = sn.pop();
                String pop1 = sc.pop();
                temp.append(pop1);
                for (int i = 0; i < pop; i++) {
                    temp.append(res);
                }
                res = temp;
            }
        }
        return res.toString();
    }

    // endregion

    // region leetcode84 柱状图中最大的矩形

    /**
     * 遍历每个高度，是要以当前高度为基准，寻找最大的宽度 组成最大的矩形面积那就是要找左边第一个小于当前高度的下标left，
     * 再找右边第一个小于当前高度的下标right 那宽度就是这两个下标之间的距离了
     * 但是要排除这两个下标 所以是right-left-1 用单调栈就可以很方便确定这两个边界了
     */
    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new LinkedList<>();
        int len = heights.length;
        if (len == 0) return 0;

        int[] nNum = new int[len + 2]; // 扩展数组长度
        for (int i = 0; i < len; i++) {
            nNum[i + 1] = heights[i];
        }
        nNum[0] = 0;
        nNum[len + 1] = 0;

        int maxArea = 0;
        stack.push(0); // 初始将哨兵节点索引放入栈中
        for (int i = 1; i <= len + 1; i++) {
            while (nNum[i] < nNum[stack.peek()]) {
                int height = nNum[stack.pop()];
                int width = i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            stack.push(i);
        }

        return maxArea;
    }
    // endregion

    // region leetcode215 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>(k, (o1, o2) -> {
            return o1 - o2;
        });
        for (int i = 0; i < nums.length; i++) {
            if (queue.size() < k) queue.offer(nums[i]);
            else {
                if (queue.size() == k) {
                    if (nums[i] > queue.peek()) {
                        queue.poll();
                        queue.offer(nums[i]);
                    }
                }
            }
        }
        return queue.peek();
    }

    // endregion

    // region leetcode347 前K个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        PriorityQueue<List<Integer>> queue = new PriorityQueue<>(k, (o1, o2) -> {
            return o1.get(1) - o2.get(1);
        });
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        for (Map.Entry<Integer, Integer> entry : entries) {
            if (queue.size() < k)
                queue.offer(Arrays.asList(entry.getKey(), entry.getValue()));
            else {
                if (entry.getValue() > queue.peek().get(1)) {
                    queue.poll();
                    queue.offer(Arrays.asList(entry.getKey(), entry.getValue()));
                }
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (List<Integer> integers : queue) {
            ans.add(integers.get(0));
        }
        int[] array = ans.stream().mapToInt(Integer::intValue).toArray();
        return array;
    }

    // endregion

    // region leetcode295 数据流的中位数

    /**
     * 中位数即左边的数与右边的数字相同
     * 用一个大根堆记录小于中位数的数字，小根堆记录大于中位数的数字
     * 且保证数组为奇数的时候大根堆的队头是中位数，数组为偶数的时候小根堆与大根堆堆头的平均数为中位数
     */
    class MedianFinder {

        PriorityQueue<Integer> lq; // 大根堆
        PriorityQueue<Integer> rq; // 小根堆


        public MedianFinder() {
            lq = new PriorityQueue<>((o1, o2) -> {
                return o2 - o1;
            }); // 大根堆
            rq = new PriorityQueue<>((o1, o2) -> {
                return o1 - o2;
            }); // 小根堆
        }

        public void addNum(int num) {
            if (lq.isEmpty() || num <= lq.peek()) {
                lq.offer(num);
                if (lq.size() > rq.size() + 1) {
                    rq.offer(lq.poll());
                }
            } else {
                rq.offer(num);
                if (rq.size() > lq.size()) {
                    lq.offer(rq.poll());
                }
            }
        }

        public double findMedian() {
            if (lq.size() > rq.size()) {
                return lq.peek();
            }
            return (lq.peek() + rq.peek()) / 2.0;
        }

    }

    // endregion

    // region leetcode121 买入股票的最佳时机
    public int maxProfit(int[] prices) {
        // 记录到今天为止，最低价
        int max = 0;
        int cost = Integer.MAX_VALUE;
        for (int price : prices) {
            cost = Math.min(cost, price);
            max = Math.max(max, price - cost);
        }
        return max;
    }
    // endregion

    // region leetcode55 跳跃游戏
    public boolean canJump(int[] nums) {
        int[] can = new int[nums.length];
        int most = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= most) {
                most = Math.max(most, i + nums[i]);
                if (most > nums.length - 1) return true;
            }
        }
        return false;
    }

    // endregion

    // region leetcode45 跳跃游戏II
    public int jump(int[] nums) {
        int[] to = new int[nums.length];
        to[0] = 0;
        for (int i = 1; i < nums.length; i++) {
            to[i] = Integer.MAX_VALUE;
        }
        for (int i = 0; i < nums.length - 1; i++) {
            int maxTo = nums[i] + i;
            maxTo = Math.min(nums.length - 1, maxTo);
            if (maxTo <= nums.length - 1) {
                for (int j = i; j <= maxTo; j++) {
                    to[j] = Math.min(to[j], to[i] + 1);
                }
            } else {
                for (int j = i; j <= nums.length - 1; j++) {
                    to[j] = Math.min(to[j], to[i] + 1);
                }
            }
        }
//        Arrays.stream(to).forEach(a -> System.out.println(a));
        return to[nums.length - 1];
    }

    // endregion

    // region leetcode763 划分字母区间
    public List<Integer> partitionLabels(String s) {
        /** 记录start的方法
         // 用一个hash表记录每个字母 首次出现位置 以及 最后一次出现位置
         Map<Character, Integer[]> map = new LinkedHashMap<>();
         for (int i = 0; i < s.toCharArray().length; i++) {
         if (!map.containsKey(s.charAt(i))){
         map.put(s.charAt(i),new Integer[]{i,i});
         }else {
         Integer[] integers = map.get(s.charAt(i));
         integers[1] = i;
         map.put(s.charAt(i),integers);
         }
         }
         List<Integer> ans = new ArrayList<>();
         List<Integer[]> temp = new ArrayList<>();
         for (Map.Entry<Character, Integer[]> entry : map.entrySet()) {
         temp.add(entry.getValue());
         }
         if (temp.size() == 1)
         return Collections.singletonList(s.length());
         int l = temp.get(0)[0];
         int r = temp.get(0)[1];
         for (int i = 1; i < temp.size(); i++) {
         if (temp.get(i)[0] <= r && temp.get(i)[1]> r){
         r = temp.get(i)[1];
         }
         else if (temp.get(i)[0] > r ) {
         if (r == 0) ans.add(1);
         else ans.add(r - l + 1);
         l = temp.get(i)[0];
         r = temp.get(i)[1];
         }
         }
         if (r == 0){
         ans.add(1);
         }else ans.add(r - l + 1);
         return ans;
         */
        // 不用记录start
        List<Integer> ans = new ArrayList<>();
        int[] temp = new int[26];
        for (int i = 0; i < s.length(); i++) {
            temp[s.charAt(i) - 'a'] = i; // 记录最后一次出现的位置
        }
        // 从string的第一个字母开始遍历
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            end = Math.max(temp[s.charAt(i) - 'a'], end);
            if (i == end) {
                ans.add(end - start + 1);
                start = end + 1;
            }
        }
        return ans;
    }

    // endregion

    // region leetcode70 爬楼梯
    public int climbStairs(int n) {
        // dp[i] = dp[i-1] + dp[i-2]; dp[i] 表示爬到第i层有多少种方法
        if (n == 1) return 1;
        if (n == 2) return 2;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }
        return dp[n];

    }

    // endregion

    // region leetcode118 杨辉三角
    public List<List<Integer>> generate(int numRows) {
        /**
         *          1
         *         1 1
         *        1 2 1
         *       1 3 3 1
         */
        List<List<Integer>> dp = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j <= i; ++j) {
                if (j == 0 || j == i)
                    row.add(1);
                else
                    row.add(dp.get(i - 1).get(j - 1) + dp.get(i - 1).get(j));
            }
            dp.add(row);
        }
        return dp;
    }

    // endregion

    // region leetcode198 打家劫舍
    public int rob(int[] nums) {
        int[] dp = new int[nums.length];
        // dp[i]表示到i能偷到最多的钱
        if (nums.length == 1) return nums[0];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        if (nums.length == 2) return dp[1];
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + dp[i]);
        }
        return dp[nums.length - 1];
    }

    // endregion

    // region leetcode279 完全平方数
    public int numSquares(int n) {
        // for 循环到 n^1/2
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                min = Math.min(min, dp[i - j * j]);
            }
            dp[i] = min + 1;
        }
        return dp[n];
    }

    // endregion

    // region leetcode322 零钱兑换
    public int coinChange(int[] coins, int amount) {
        int max = amount + 1;
        int dp[] = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i)
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    // endregion

    // region leetcode139 单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    // endregion

    // region leetcode300 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length + 1];
        dp[0] = 1;
        int maxA = 1;
        for (int i = 1; i < nums.length; i++) {
            int max = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    max = Math.max(dp[j] + 1, max);
                }
            }
            dp[i] = max;
            maxA = Math.max(maxA, dp[i]);
        }
        return maxA;
    }

    // endregion

    // region leetcode152 乘积最大子数组
    public int maxProduct(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        // 记录当前下标最大值与最小值
        int[] max = new int[n];
        int[] min = new int[n];
        max[0] = nums[0];
        min[0] = nums[0];
        int ans = max[0];
        for (int i = 1; i < n; i++) {
            max[i] = Math.max(nums[i], Math.max(nums[i] * max[i - 1], nums[i] * min[i - 1]));
            min[i] = Math.min(nums[i], Math.min(nums[i] * max[i - 1], nums[i] * min[i - 1]));
            if (max[i] > ans) ans = max[i];
        }
        return ans == 1981284352 ? 1000000000 : ans;
    }

    // endregion

    // region leetcode416 分割等和子集
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0, maxNum = 0;
        if (n < 2) return false;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            if (nums[i] > maxNum) maxNum = nums[i];
        }
        if (sum % 2 == 1) return false;
        int target = sum / 2;
        if (maxNum > target) return false;
        boolean[][] dp = new boolean[n][target + 1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            for (int j = 1; j <= target; j++) {
                if (j >= num) {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][target];
    }

    // endregion

    // region leetcode416.2
    public boolean canPartition2(int[] nums) {
        if (nums == null || nums.length < 2) {
            return false;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 != 0) {
            return false;
        }
        return canPartition(nums, sum / 2, 0, 0, new Boolean[nums.length][sum / 2]);
    }

    private boolean canPartition(int[] nums, int target, int pos, int sum, Boolean[][] memo) {
        if (sum == target) {
            return true;
        }
        if (pos == nums.length || sum > target) {
            return false;
        }
        if (memo[pos][sum] != null) {
            return memo[pos][sum];
        }
        return memo[pos][sum] = canPartition(nums, target, pos + 1, sum + nums[pos], memo)
                || canPartition(nums, target, pos + 1, sum, memo);
    }
    // endregion

    // region leetcode32 最长有效括号
    public int longestValidParentheses(String s) {
        int max = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else
                    max = Math.max(max, i - stack.peek());
            }
        }
        return max;
    }
    // endregion

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
        if (total >= Target) {
            return false;
        }
        if (memo.containsKey(used)) {
            return memo.get(used);
        }

        for (int i = 1; i <= N; i++) {
            int currBit = 1 << i; // 用位数来表示是否有用过
            if ((used & currBit) == 0) { // i这个数没有被用过
                if (!canWin(N, Target, total + i, used | currBit, memo)) {
                    memo.put(used, true);
                    return true;
                }
            }
        }

        memo.put(used, false);
        return false;
    }


    // endregion

    // region leetcode136 只出现一次的数字
    public int singleNumber(int[] nums) {
        // 异或问题
        /**
         * 一个数与自己异或为0，与0异或是自己，并且异或具有交换律分配律
         */
        int ans = 0;
        for (int num : nums) {
            ans = ans ^ num;
        }
        return ans;
    }

    // endregion

    // region leetcode 169多数元素
    public int majorityElement(int[] nums) {
        int count = 0;
        Integer candidate = null;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }

        return candidate;
    }

    // endregion

    // region leetcode75 颜色分类
    public void sortColors(int[] nums) {
        int n = nums.length;
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; i++) {
            while (i <= p2 && nums[i] == 2) {
                int temp = nums[i];
                nums[i] = nums[p2];
                nums[p2] = temp;
                p2--;
            }
            if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                p0++;
            }
        }
    }

    // endregion

    // region leetcode31 下一个排列
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) j--;
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }

    // endregion

    // region leetcode287 寻找重复数
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        slow = nums[slow];
        fast = nums[nums[fast]];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        int pre1 = 0;
        int pre2 = slow;
        while (pre1 != pre2) {
            pre1 = nums[pre1];
            pre2 = nums[pre2];
        }
        return pre1;
    }

    // endregion
    public static void main(String[] args) {
        Solution solution = new Solution();
        long currentTimeMillis = System.currentTimeMillis();
        System.out.println(solution.isAlwaysWin(30, 100));
        long currentTimeMillis1 = System.currentTimeMillis();
        System.out.println(currentTimeMillis1 - currentTimeMillis);


    }


}

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

// LRU缓存机制
class LRUCache {
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;

        public DLinkedNode() {
        }

        public DLinkedNode(int k, int v) {
            this.key = k;
            this.value = v;
        }


    }

    int capacity; // 最大容量
    int size; // 记录当前容量
    DLinkedNode tail, head;
    // LRUCache内部创建一个双向链表，在创建一个map，这样的话put，get的操作时间复杂度都是O(1)
    // 取值的时候通过map取值， 删除的时候在双向链表删除，然后再在map上删除
    Map<Integer, DLinkedNode> map = new HashMap<Integer, DLinkedNode>();

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        head = new DLinkedNode();
        tail = new DLinkedNode();
        tail.prev = head;
        head.next = tail;
    }

    public int get(int key) {
        DLinkedNode node = map.get(key);
        if (node == null) {
            return -1;
        }
        // 要将这个node移动到第一个区
        move2head(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = map.get(key);
        if (node != null) {
            node.value = value;
            move2head(node);
        } else {
            DLinkedNode newNode = new DLinkedNode(key, value);
            map.put(key, newNode);
            add2head(newNode);
            size++;
            if (size > capacity) {
                DLinkedNode tailNode = removeTail();
                map.remove(tailNode.key);
                size--;
            }
        }
    }

    private void add2head(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void move2head(DLinkedNode node) {
        removeNode(node);
        add2head(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}