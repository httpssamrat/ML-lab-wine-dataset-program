from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def run_models(df):
    X = df.drop('Wine_Class', axis=1)
    y = df['Wine_Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=4)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
    print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
