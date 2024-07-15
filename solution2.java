import java.util.HashMap;

public class solution2 {
    class LRUCache {
        class DNode {
            int key;
            int value;
            DNode pre;
            DNode next;
            public DNode(int key ,int value) {
                this.key = key;
                this.value = value;
            }
            public DNode() {
            }
        }

        int size;
        int capacity;
        DNode tail, head;

        HashMap<Integer, DNode> map = new HashMap<>();
        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            head = new DNode();
            tail = new DNode();
            head.next = tail;
            tail.pre = head;
        }

        public int get(int key) {
            DNode node = map.get(key);
            if (node==null)
                return -1;
            // 将这个node移动到第一个去
            move2head(node);
            return node.value;
        }

        public void put(int key, int value) {
            DNode node = map.get(key);
            if (node !=null){
                node.value = value;
                move2head(node);
            }else {
                DNode newNode = new DNode(key, value);
                map.put(key, newNode);
                add2Node(newNode);
                size++;
                if (size>capacity){
                    DNode tailNode = removeTail();
                    map.remove(tailNode.key);
                    size--;
                }
            }
        }
        public void removeNode(DNode node){
            node.pre.next = node.next;
            node.next.pre = node.pre;
        }
        public void add2Node(DNode node){
            node.pre = head;
            node.next = head.next;
            head.next.pre = node;
            head.next = node;
        }
        public void move2head(DNode node) {
            removeNode(node);
            add2Node(node);
        }
        public DNode removeTail(){
            DNode target = tail.pre;
            removeNode(target);
            return target;
        }
    }
}
