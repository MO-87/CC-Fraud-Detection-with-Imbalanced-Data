from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def Base_model(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def RFC_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=15, random_state=42)
    model.fit(X_train, y_train)
    return model


def NN_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32),
                          max_iter=300,
                          random_state=42,
                          solver='adam',
                          activation='relu')
    model.fit(X_train, y_train)
    return model


def XGB_model(X_train, y_train):
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=20,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def Voting_model(X_train, y_train):
    model = VotingClassifier(
            estimators=[('NN', NN_model(X_train, y_train)),
                        ('RF', RFC_model(X_train, y_train)),
                        ('XGB', XGB_model(X_train, y_train))],
            voting='soft',
            weights=[1, 1.5, 2])

    model.fit(X_train, y_train)
    return model
