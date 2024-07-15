public class shunxuExecute {
    public static void main(String[] args) {
        Work work = new Work();
        Thread thread = new Thread(()->{
            try {
                work.subThreadLoop();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        thread.start();
        try {
            work.mainThreadLoop();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}

class Work {

    boolean flag = true;
    public synchronized void subThreadLoop() throws InterruptedException {
        for (int i = 1; i <= 5; i++) {
            while (flag == false){
                wait();
            }
            for (int j = 1; j <= 10; j++) {
                System.out.println("sub thread:" + j );
                Thread.sleep(100);
            }
            notify();
            wait();
        }
    }

    public synchronized void mainThreadLoop() throws InterruptedException {
        for (int i = 1; i <= 5; i++) {
            while (flag == true){
                wait();
            }
            for (int j = 1; j <= 10; j++) {
                System.out.println("main thread:" + j );
                Thread.sleep(100);
            }
            notify();
            wait();
        }
    }
}
