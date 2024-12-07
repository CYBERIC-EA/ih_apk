from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.recycleview import RecycleView
from kivy.uix.popup import Popup
from kivy.uix.recyclegridlayout import RecycleGridLayout
## Import libraries for data preprocessing
# For Data Analysis
import pandas as pd
import numpy as np 
# For Data Visualization
import seaborn as sns
import collections as Counter
# For Feature Importance for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve
import pandas as pd
import numpy as np
import sys, os
# Load dataset
# Get the base path where the program is running
if getattr(sys, 'frozen', False):  # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:  # Running in a normal Python environment
    base_path = os.path.abspath(".")

# Construct the path to the CSV file
csv_file = os.path.join(base_path, "IHD Dataset.csv")

# Load the CSV
try:
    data = pd.read_csv(csv_file)
    print("Data loaded successfully:")
    print(data.head())
except FileNotFoundError:
    print("Default data file not found.")

#data=pd.read_csv(r"IHD Dataset.csv")


def predict_IHD(input_data, data = data):
    # ### Data Cleaning
    # This is to ensure consistency in the data. The glucose column will be dropped first as it is not needed in the prediction.

    # to drop the irrelevent column
    data=data.drop(["Glucose","Education"], axis=1)
    # to check for missing value
    data.isnull().sum()

    # to fill the missing values
    #for Numerical columns
    data['CigarettesPerDay'] = data['CigarettesPerDay'].fillna(data['CigarettesPerDay'].median())
    data['BMI'] = data['BMI'].fillna(data['BMI'].median())
    data['HeartRate'] = data['HeartRate'].fillna(data['HeartRate'].median())

    #for Categorical columns
    data['BPmeds'] = data['BPmeds'].fillna(data['BPmeds'].mode()[0])
    data['BPmeds'] = data['BPmeds'].astype(int)


    data.isnull().sum()

    # check for duplicates
    data.duplicated().sum()

    ### Treating Outliers
    # The outliers in this data will be filtered out by capping the values in the columns to a certain threshold based on domain knowledge.

    print(data.shape)

    # Define the thresholds for each feature
    thresholds = {
        'Systolic BP': (60, 180),
        'DiastolicBP': (40, 120),
        'HeartRate': (50, 120),
        'BMI': (17, 35),
        'CigarettesPerDay': (0, 50)
    }

    # Apply the thresholds to filter out outliers
    for feature, (lower, upper) in thresholds.items():
        if feature in data.columns:
            data = data[(data[feature] >= lower) & (data[feature] <= upper)]

    # Check the resulting dataset after filtering
    print(data.shape)


    ## Modelling and Feature Engineering
    # Data Pre-Processing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    # Evaluation metrics
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    features=data[['Sex', 'Age', 'CurrentSmoker', 'CigarettesPerDay', 'BPmeds', 'PrevalentStroke', 'PrevalentHypertension',
                'Diabetes', 'Systolic BP', 'DiastolicBP', 'BMI', 'HeartRate']]
    target=data['TenYearCHD']
    features.head(2)
    target.head(2)

    X = features
    y = target


    Xtrain,Xtest,ytrain,ytest = train_test_split(features, target, test_size = 0.2, random_state = 42)


    # ## Model training before scaling the data
    # The modeling was conducted on unscaled data without the application of any advanced techniques to enhance the results. Despite this, the XGBoost, Naive Bayes, K-Nearest Neighbours and Decision Tree models demonstrated superior performance compared to the other models, making them suitable for further analysis. I will be selecting these four models for the next phase of analysis, where I intend to scale the data prior to training to improve the results and incorporate other advanced techniques to optimize their performance and enhance prediction accuracy.


    classifiers = [ 
        [LogisticRegression(), 'Logistic Regression']  
    ]


    for classifier in classifiers:
        model = classifier[0]  # Get the model
        model_name = classifier[1]  # Get the name of the model
        print(f"<---{model_name}--->")

        try:
            model.fit(Xtrain, ytrain)

            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                prob_predictions = model.predict_proba(Xtest)[:, 1]  # Get probability of the positive class
                
                # Calculate AUC-ROC score
                auc_score = roc_auc_score(ytest, prob_predictions)
                print(f"AUC-ROC Score: {auc_score:.5f}")
            else:
                print(f"{model_name} does not support probability predictions.")
                auc_score = None 

            predictions = model.predict(Xtest)
            report = classification_report(ytest, predictions, digits=5)
            matrix = confusion_matrix(ytest, predictions)
            print(report)
            

        except Exception as e:
            print(f"An error occurred with {model_name}: {e}")



    # ## Improve model performance by Scaling
    # From the exploratory data analysis performed, it showed that the data is normally distributed. In this case, the data will be standardized by using the StandardScaler


    from sklearn.preprocessing import RobustScaler


    scaler=RobustScaler()


    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.fit_transform(Xtest)


    Xtrain_scaled_df = pd.DataFrame(Xtrain_scaled, columns=Xtrain.columns)
    print(Xtrain_scaled_df.describe())


    for classifier in classifiers:
        model = classifier[0]
        model_name = classifier[1]
        print(f"<---{model_name}--->")
        
        # Train the model
        model.fit(Xtrain_scaled, ytrain)
        
        # Generate predictions
        predictions = model.predict(Xtest_scaled)
        report = classification_report(ytest, predictions, digits=5)
        
        # Calculate AUC-ROC if possible
        try:
            if hasattr(model, "predict_proba"):
                # Use predict_proba for AUC-ROC calculation
                prob_predictions = model.predict_proba(Xtest_scaled)[:, 1]
                auc_score = roc_auc_score(ytest, prob_predictions)
            elif hasattr(model, "decision_function"):
                # Use decision_function for AUC-ROC calculation
                decision_scores = model.decision_function(Xtest_scaled)
                auc_score = roc_auc_score(ytest, decision_scores)
            else:
                auc_score = None
                print(f"{model_name} does not support AUC-ROC calculation with predict_proba or decision_function.")
            
            if auc_score is not None:
                print(f"AUC-ROC Score: {auc_score:.5f}")
                
        except Exception as e:
            print(f"Error calculating AUC-ROC for {model_name}: {e}")

        # Print classification report
        print(report)
        



    # ## Model Improvement by adjusting Class Weights
    # After scaling my data, I noticed a little improvement in the results. I will now improve the model to see if this will improve the performance and result. I'll give class 1 more importance than class 0 by increasing the weight importance.


    # ### Logistic Regression Class Weight


    logistic_model = LogisticRegression(class_weight={0:1, 1:5}, random_state=42)
    logistic_model.fit(Xtrain_scaled, ytrain)

    logistic_predictions = logistic_model.predict(Xtest_scaled)

    logistic_prob_predictions = logistic_model.predict_proba(Xtest_scaled)[:, 1]

    logistic_auc_roc = roc_auc_score(ytest, logistic_prob_predictions)
    print(f"Logistic Regression AUC-ROC Score: {logistic_auc_roc:.5f}")

    logistic_report = classification_report(ytest, logistic_predictions, digits=5)
    print("Logistic Regression Classification Report:\n", logistic_report)




    ### Logistic Regression Cross Validation

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
    import numpy as np

    # Ensure ytrain is a NumPy array
    ytrain = np.array(ytrain)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []
    precision_scores = []
    accuracy_scores = []
    recall_scores = []
    f1_scores = []

    print("<--- Logistic Regression Classifier --->")

    for fold, (train_index, test_index) in enumerate(kfold.split(Xtrain_scaled, ytrain), 1):
        # Correctly split the data using NumPy array indexing
        X_train_fold, X_test_fold = Xtrain_scaled[train_index], Xtrain_scaled[test_index]
        y_train_fold, y_test_fold = ytrain[train_index], ytrain[test_index]

        model = LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)[:, 1]

        roc_auc = roc_auc_score(y_test_fold, y_proba)
        precision = precision_score(y_test_fold, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test_fold, y_pred)
        recall = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)

        roc_auc_scores.append(roc_auc)
        precision_scores.append(precision)
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold} - ROC-AUC: {roc_auc:.5f}, Precision: {precision:.5f}, Accuracy: {accuracy:.5f}, Recall: {recall:.5f}, F1 Score: {f1:.5f}")

    print("\nAverage Scores for Logistic Regression Classifier")
    print(f"Mean ROC-AUC Score: {np.mean(roc_auc_scores):.5f}")
    print(f"Mean Precision Score: {np.mean(precision_scores):.5f}")
    print(f"Mean Accuracy Score: {np.mean(accuracy_scores):.5f}")
    print(f"Mean Recall Score: {np.mean(recall_scores):.5f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.5f}")
    print("\n" + "-"*50 + "\n")


    ## Cross Validation and Hyperparameter Tuning for Logistic Regression


    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve
    import numpy as np

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(Xtrain_scaled, ytrain, test_size=0.2, random_state=42, stratify=ytrain)

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Regularization types
        'solver': ['liblinear'],  # Optimization solvers
        'class_weight': [{0: 1, 1: w} for w in [1, 2, 5]] 
    }

    # Set up cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use F1-score as the scoring metric
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid,
        scoring='f1',  # Optimize for F1-score
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    # Perform hyperparameter tuning
    print("<--- Hyperparameter Tuning --->")
    grid_search.fit(X_train, y_train)

    # Extract best hyperparameters and best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the validation set
    y_proba = best_model.predict_proba(X_val)[:, 1]

    # Use the Precision-Recall Curve to find the optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)  # Maximize F1-score
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.2f}")

    # Apply the optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Calculate metrics on the validation set
    roc_auc = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Print metrics
    print("\nMetrics for Best Logistic Regression Model")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print("\n" + "-"*50 + "\n")


    # Feature Importance for Logistic Regression

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(Xtrain_scaled, ytrain, test_size=0.2, random_state=42, stratify=ytrain)

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Regularization types
        'solver': ['liblinear'],  # Optimization solvers
        'class_weight': [{0: 1, 1: w} for w in [1, 2, 5]] 
    }

    # Set up cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use F1-score as the scoring metric
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid,
        scoring='f1',  # Optimize for F1-score
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    # Perform hyperparameter tuning
    print("<--- Hyperparameter Tuning --->")
    grid_search.fit(X_train, y_train)

    # Extract best hyperparameters and best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the validation set
    y_proba = best_model.predict_proba(X_val)[:, 1]

    # Use the Precision-Recall Curve to find the optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)  # Maximize F1-score
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.2f}")

    # Apply the optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Calculate metrics on the validation set
    roc_auc = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Print metrics
    print("\nMetrics for Best Logistic Regression Model")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print("\n" + "-"*50 + "\n")
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data (same scaler used during training)
    input_scaled = scaler.transform(input_df)

    # Predict probability for ischemic heart disease (IHD)
    predicted_proba = best_model.predict_proba(input_scaled)[:, 1]

    # Apply the optimal threshold to predict the class (IHD or not)
    predicted_class = (predicted_proba >= optimal_threshold).astype(int)

    print(f"Predicted Probability of IHD: {predicted_proba[0]:.4f}")
    print(f"Predicted Class: {'IHD, See your Doctor immediately' if predicted_class[0] == 1 else 'No IHD, Maintain a healthy lifestyle and monitor your health regularly.'}")
    return f" { 'IHD, See your Doctor immediately' if predicted_class[0] == 1 else 'No IHD, Maintain a healthy lifestyle and monitor your health regularly.'} "

class FeatureApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10

        # Define feature names
        self.features = [
            "Sex", "Age", "Current Smoker", "Cigarettes Per Day",
            "BP Meds", "Prevalent Stroke", "Prevalent Hypertension",
            "Diabetes", "Systolic BP", "Diastolic BP", "BMI", "Heart Rate"
        ]

        # Input fields layout
        input_layout = GridLayout(cols=2, size_hint_y=None, height=400)
        self.inputs = {}

        for feature in self.features:
            input_layout.add_widget(Label(text=feature))
            self.inputs[feature] = TextInput(multiline=False)
            input_layout.add_widget(self.inputs[feature])

        self.add_widget(input_layout)

        # Buttons layout
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        self.add_button = Button(text="Add Feature")
        self.add_button.bind(on_press=self.add_feature)
        button_layout.add_widget(self.add_button)

        self.delete_button = Button(text="Delete Feature")
        self.delete_button.bind(on_press=self.delete_feature)
        button_layout.add_widget(self.delete_button)

        self.predict_button = Button(text="Predict")
        self.predict_button.bind(on_press=self.predict_feature)
        button_layout.add_widget(self.predict_button)

        self.add_widget(button_layout)

        # Table (RecycleView) layout
        self.table = RecycleView(size_hint=(1, None), height=150)
        self.table.layout_manager = RecycleGridLayout(
            cols=len(self.features), 
            default_size=(None, 50), 
            default_size_hint=(1, None), 
            size_hint_y=None, 
            height=50
        )
        self.table.viewclass = 'Label'
        self.table.add_widget(self.table.layout_manager)
        self.add_widget(self.table)

    def add_feature(self, instance):
        # Collect feature values
        values = [self.inputs[feature].text for feature in self.features]

        if all(values):  # Ensure all fields are filled
            # Replace row if one exists
            self.table.data = [{"text": val} for val in values]
            for feature in self.features:
                self.inputs[feature].text = ""  # Clear the input fields
        else:
            popup = Popup(title="Input Error", content=Label(text="Please fill in all fields."), size_hint=(0.6, 0.4))
            popup.open()

    def delete_feature(self, instance):
        if self.table.data:
            self.table.data = []
        else:
            popup = Popup(title="Delete Error", content=Label(text="No feature to delete."), size_hint=(0.6, 0.4))
            popup.open()

    def predict_feature(self, instance):
        if self.table.data:
            # Rebuild the feature dictionary
            values = [item["text"] for item in self.table.data]
            input_data = {
                'Sex': int(values[0]),  # Assuming 'Sex' is either 1 (Male) or 0 (Female)
                'Age': float(values[1]),
                'CurrentSmoker': int(values[2]),  # Assuming 1 (Yes) or 0 (No)
                'CigarettesPerDay': float(values[3]),
                'BPmeds': int(values[4]),  # 1 (Yes) or 0 (No)
                'PrevalentStroke': int(values[5]),  # 1 (Yes) or 0 (No)
                'PrevalentHypertension': int(values[6]),  # 1 (Yes) or 0 (No)
                'Diabetes': int(values[7]),  # 1 (Yes) or 0 (No)
                'Systolic BP': float(values[8]),
                'DiastolicBP': float(values[9]),
                'BMI': float(values[10]),
                'HeartRate': float(values[11])
            }

            # Example logic for prediction
            result = predict_IHD(input_data)
            self.show_prediction(result)
        else:
            popup = Popup(title="Predict Error", content=Label(text="No features to predict."), size_hint=(0.6, 0.4))
            popup.open()

    def show_prediction(self, input_data):
        popup = Popup(title="Prediction",
                      content=Label(text=f"Prediction:\n{input_data}"),
                      size_hint=(0.8, 0.6))
        popup.open()


class FeatureAppKivy(App):
    def build(self):
        return FeatureApp()


if __name__ == '__main__':
    FeatureAppKivy().run()
