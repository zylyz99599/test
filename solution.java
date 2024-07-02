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

    // 合并K个有序链表
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


    public static void main(String[] args) {
        Solution solution = new Solution();
        ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        System.out.println(solution.isPalindrome(l1));


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
        if (node==null){
            return -1;
        }
        // 要将这个node移动到第一个区
        move2head(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = map.get(key);
        if (node!=null){
            node.value = value;
            move2head(node);
        }else {
            DLinkedNode newNode = new DLinkedNode(key,value);
            map.put(key,newNode);
            add2head(newNode);
            size++;
            if (size>capacity){
                DLinkedNode tailNode = removeTail();
                map.remove(tailNode.key);
                size--;
            }
        }
    }
    private void add2head(DLinkedNode node){
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node){
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void move2head(DLinkedNode node){
        removeNode(node);
        add2head(node);
    }
    private DLinkedNode removeTail(){
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}
