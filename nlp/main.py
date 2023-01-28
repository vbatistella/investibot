from model import Model

USEBACKUP = False
TRAIN = True

def main():
    model = Model(1)
    model_name = "investibot_6m_to_1m"

    if TRAIN:
        # for i in range(100):
        print("Getting data...")
        x_train, y_train, x_test, y_test = 1,1,1,1 # DATA HERE
        print("Training...")
        model.train(x_train, y_train, 100, 128)
        print(model.performance(x_test, y_test))
        model.save(model_name)
    else:
        print("Loading...")
        model.load(model_name)
    
    # data.show(model, "googl.us.txt")

if __name__ == "__main__":
    main()