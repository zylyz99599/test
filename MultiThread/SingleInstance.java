package MultiThread;

public class SingleInstance {
        private static final SingleInstance INSTANCE = new SingleInstance();

        private SingleInstance(){};

        public static SingleInstance getInstance(){
            return INSTANCE;
        }
}
