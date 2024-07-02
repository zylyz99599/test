package MultiThread;

public class LazeModal {

    private static volatile LazeModal instance = null;

    private LazeModal(){};

    public static LazeModal getInstance(){
        if (instance == null){
            synchronized (LazeModal.class){
                if (instance == null)
                    instance = new LazeModal();
            }
        }
        return instance;
    }
}
