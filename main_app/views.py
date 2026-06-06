from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import tensorflow as tf


# ---------------- SIGNUP ----------------
def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("signup")

        user = User.objects.create_user(username=username, password=password)
        user.save()

        messages.success(request, "Account created successfully")
        return redirect("login")

    return render(request, "signup.html")


# ---------------- LOGIN ----------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid credentials")

    return render(request, "login.html")


# ---------------- DASHBOARD ----------------
def dashboard_view(request):
    if not request.user.is_authenticated:
        return redirect("login")

    return render(request, "dashboard.html")


# ---------------- LOGOUT ----------------
def logout_view(request):
    logout(request)
    return redirect("login")







# project1

from django.shortcuts import render
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def city(request):
    result = None
    best_model_name = None
    best_accuracy = 0
    error = None
    model_results = {}
    input_data_display = {}

    if request.method == "POST":
        try:
            # Get form data
            city_name = request.POST.get("city")
            co = float(request.POST.get("co"))
            no2 = float(request.POST.get("no2"))
            so2 = float(request.POST.get("so2"))
            o3 = float(request.POST.get("o3"))
            pm25 = float(request.POST.get("pm25"))
            pm10 = float(request.POST.get("pm10"))

            # Store input for UI
            input_data_display = {
                "City": city_name,
                "CO": co,
                "NO2": no2,
                "SO2": so2,
                "O3": o3,
                "PM2.5": pm25,
                "PM10": pm10
            }

            # Load dataset
            df = pd.read_csv(r"E:\Internship_project\Dataset\Air-quality-Classification\City_Types.csv")

            df = df.drop(columns=["Date"])

            # Encode target
            le = LabelEncoder()
            df["Type"] = le.fit_transform(df["Type"])

            X = df[["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]]
            y = df["Type"]

            if y.nunique() < 2:
                error = "Dataset must have at least 2 classes"
                return render(request, "city.html", {"error": error})

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            best_model = None

            # Train & compare
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                model_results[name] = round(acc * 100, 2)

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model_name = name
                    best_model = model

            # Prediction
            input_scaled = scaler.transform([[co, no2, so2, o3, pm25, pm10]])
            prediction = best_model.predict(input_scaled)

            result = le.inverse_transform(prediction)[0]

        except Exception as e:
            error = str(e)

    return render(request, "city.html", {
        "result": result,
        "best_model": best_model_name,
        "accuracy": round(best_accuracy * 100, 2),
        "model_results": model_results,
        "input_data": input_data_display,
        "error": error
    })





#project2
from django.shortcuts import render
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def plant(request):
    result = None
    best_model_name = None
    best_accuracy = 0
    model_results = {}
    error = None
    confidence = None
    input_summary = {}

    def get_float(value):
        try:
            return float(value)
        except:
            return 0

    if request.method == "POST":
        try:
            # ===== INPUT =====
            lv = get_float(request.POST.get("vibration"))
            pollen = get_float(request.POST.get("scent"))
            bio = get_float(request.POST.get("light"))
            root = get_float(request.POST.get("root"))
            growth = get_float(request.POST.get("growth"))
            temp = get_float(request.POST.get("temp"))
            moisture = get_float(request.POST.get("moisture"))
            sunlight = get_float(request.POST.get("sunlight"))
            fungus = int(request.POST.get("fungus") or 0)

            # ===== LOAD DATASET =====
            df = pd.read_csv(r"E:\Internship_project\Dataset\Plant-communication-classification\plant_data.csv")
            df = df.dropna()

            df.columns = [
                "vibration", "scent", "light", "root",
                "growth", "temp", "moisture",
                "sunlight", "fungus", "target"
            ]

            # ===== ENCODE TARGET =====
            le = LabelEncoder()
            df["target"] = le.fit_transform(df["target"])

            X = df.drop(columns=["target"])
            y = df["target"]

            # ===== SPLIT =====
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ===== MODELS =====
            models = {

                "KNN": GridSearchCV(
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("knn", KNeighborsClassifier())
                    ]),
                    {
                        "knn__n_neighbors": [3, 5, 7],
                        "knn__weights": ["uniform", "distance"]
                    },
                    cv=5
                ),

                "SVM": GridSearchCV(
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("svm", SVC(probability=True))
                    ]),
                    {
                        "svm__C": [1, 10],
                        "svm__kernel": ["linear", "rbf"]
                    },
                    cv=5
                ),

                "Decision Tree": GridSearchCV(
                    DecisionTreeClassifier(),
                    {
                        "max_depth": [5, 10],
                        "min_samples_split": [2, 5]
                    },
                    cv=5
                ),

                "Random Forest": GridSearchCV(
                    RandomForestClassifier(),
                    {
                        "n_estimators": [100],
                        "max_depth": [10, None]
                    },
                    cv=5
                ),

                "Logistic Regression": GridSearchCV(
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=1000))
                    ]),
                    {
                        "lr__C": [0.1, 1, 10]
                    },
                    cv=5
                )
            }

            best_model = None

            # ===== TRAIN =====
            for name, model in models.items():
                model.fit(X_train, y_train)
                best_est = model.best_estimator_

                y_pred = best_est.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                model_results[name] = round(acc * 100, 2)

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model_name = name
                    best_model = best_est

            # ===== PREDICTION =====
            input_df = pd.DataFrame([{
                "vibration": lv,
                "scent": pollen,
                "light": bio,
                "root": root,
                "growth": growth,
                "temp": temp,
                "moisture": moisture,
                "sunlight": sunlight,
                "fungus": fungus
            }])

            prediction = best_model.predict(input_df)
            result = le.inverse_transform(prediction)[0]

            # ===== CONFIDENCE =====
            try:
                if hasattr(best_model, "predict_proba"):
                    probs = best_model.predict_proba(input_df)
                    confidence = round(max(probs[0]) * 100, 2)
            except:
                confidence = None

            # ===== INPUT SUMMARY =====
            input_summary = {
                "Leaf Vibration": lv,
                "Pollen Scent": pollen,
                "Light Intensity": bio,
                "Root Signal": root,
                "Growth Rate": growth,
                "Temperature": temp,
                "Moisture": moisture,
                "Sunlight": sunlight,
                "Fungus": fungus
            }

        except Exception as e:
            error = str(e)

    return render(request, "plant.html", {
        "result": result,
        "best_model": best_model_name,
        "accuracy": round(best_accuracy * 100, 2),
        "model_results": model_results,
        "confidence": confidence,
        "input_summary": input_summary,
        "error": error
    })


 
 
 # Project3
 
from django.shortcuts import render
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load files
model = load_model(r"E:\Internship_project\my_ai_project\my_ai_project\crop\crop-training\fertilizer_model.h5")
scaler = pickle.load(open(r"E:\Internship_project\my_ai_project\my_ai_project\crop\crop-training\scaler.pkl", "rb"))
encoders = pickle.load(open(r"E:\Internship_project\my_ai_project\my_ai_project\crop\crop-training\encoders.pkl", "rb"))
target_encoder = pickle.load(open(r"E:\Internship_project\my_ai_project\my_ai_project\crop\crop-training\target_encoder.pkl", "rb"))

def detect(request):
    result = None
    confidence = None
    tips = []
    input_summary = {}
    error = None

    try:
        if request.method == "POST":

            data = {
                "Soil_Type": request.POST.get("soil_type"),
                "Soil_pH": float(request.POST.get("soil_ph")),
                "Soil_Moisture": float(request.POST.get("soil_moisture")),
                "Nitrogen_Level": float(request.POST.get("nitrogen")),
                "Phosphorus_Level": float(request.POST.get("phosphorus")),
                "Potassium_Level": float(request.POST.get("potassium")),
                "Temperature": float(request.POST.get("temperature")),
                "Humidity": float(request.POST.get("humidity")),
                "Crop_Type": request.POST.get("crop_type"),
            }

            input_summary = data.copy()

            # Encode categorical
            for col in encoders:
                if col in data:
                    try:
                        data[col] = encoders[col].transform([data[col]])[0]
                    except:
                        data[col] = 0

            input_data = np.array(list(data.values())).reshape(1, -1)
            input_data = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)

            result = target_encoder.inverse_transform([predicted_class])[0]
            confidence = round(np.max(prediction) * 100, 2)

            # =========================
            # SMART TIPS (RULE-BASED)
            # =========================
            if input_summary["Soil_pH"] < 6:
                tips.append("Soil is acidic → consider lime treatment.")
            elif input_summary["Soil_pH"] > 7.5:
                tips.append("Soil is alkaline → use organic compost.")

            if input_summary["Soil_Moisture"] < 0.3:
                tips.append("Low moisture → increase irrigation.")
            elif input_summary["Soil_Moisture"] > 0.8:
                tips.append("High moisture → improve drainage.")

            if input_summary["Nitrogen_Level"] < 50:
                tips.append("Low nitrogen → nitrogen-rich fertilizer needed.")

            if input_summary["Temperature"] > 35:
                tips.append("High temperature → monitor crop stress.")

    except Exception as e:
        error = str(e)

    return render(request, "detect.html", {
        "result": result,
        "confidence": confidence,
        "tips": tips,
        "input_summary": input_summary,
        "error": error
    })