from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X, y):
    """Train multiple models and select the best one based on accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (K-Nearest Neighbors)": KNeighborsClassifier(),
        "SVM (Support Vector Machine)": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    print("\nTraining models...")
    for name, model in models.items():
        # Tree models don't need scaling, distance models do
        if name in ["Decision Tree", "Random Forest"]:
            X_tr, X_te = X_train, X_test
        else:
            X_tr, X_te = X_train_scaled, X_test_scaled

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Model_Object": model
        })
        print(f" {name:<35} Accuracy: {acc*100:6.2f}%")

    # Sort results to find the best model
    results.sort(key=lambda x: x["Accuracy"], reverse=True)
    best_model_info = results[0]
    
    return (
        best_model_info['Model_Object'], 
        best_model_info['Model'], 
        scaler, 
        best_model_info['Accuracy'], 
        best_model_info['F1-Score']
    )
