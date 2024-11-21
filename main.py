from data_utils import *
from models_training_utils import *
from models_eval_utils import *


if __name__ == "__main__":

    X_train, y_train = data_loader('Data/train.csv')
    # print(X_train.head(5))
    X_val, y_val = data_loader('Data/val.csv')
    X_test, y_test = data_loader('Data/test.csv')

    X_train, X_val, scaler = data_preprocessing(X_train, X_val)
    X_test = scaler.transform(X_test)


    # X_train, y_train = under_sampling(X_train, y_train)
    X_train_resampled, y_train_resampled = over_sampling(X_train, y_train)

    choice = 1
    while choice:
        choice = int(input("1. Logistic Regression\n"
                           "2. Random Forest Classifier\n"
                           "3. Neural Network\n"
                           "4. XGBoost Classifier\n"
                           "5. Voting Classifier \n"
                           "0. Exit\n"
                           "Choose Model:"))
        if choice == 1:
            print("\n\nLogistic Regression Results:")
            Base_Model = Base_model(X_train, y_train)
            model_eval(Base_Model, X_train, y_train, "Train")
            model_eval(Base_Model, X_val, y_val, "Val")
            model_eval(Base_Model, X_test, y_test, "Test")

        elif choice == 2:
            print("\n\nRandom Forest Classifier Results:")
            RFC_Model = RFC_model(X_train, y_train)
            model_eval(RFC_Model, X_train, y_train, "Train")
            model_eval(RFC_Model, X_val, y_val, "Val")
            model_eval(RFC_Model, X_test, y_test, "Test")

        elif choice == 3:
            print("\n\nNeural Network Results:")
            NN_Model = NN_model(X_train, y_train)
            model_eval(NN_Model, X_train, y_train, "Train")
            model_eval(NN_Model, X_val, y_val, "Val")
            model_eval(NN_Model, X_test, y_test, "Test")

        elif choice == 4:
            print("\n\nXGBoost Classifier Results:")
            XGB_Model = XGB_model(X_train, y_train)
            model_eval(XGB_Model, X_train, y_train, "Train")
            model_eval(XGB_Model, X_val, y_val, "Val")
            model_eval(XGB_Model, X_test, y_test, "Test")
        elif choice == 5:
            print("\n\nVoting Classifier Results:")
            Voting_Model = Voting_model(X_train, y_train)
            model_eval(Voting_Model, X_train, y_train, "Train")
            model_eval(Voting_Model, X_val, y_val, "Val")
            model_eval(Voting_Model, X_test, y_test, "Test")
