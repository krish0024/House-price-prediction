# 🏠 House Price Prediction API (Streamlit)

This project predicts **house prices** using a simple **Linear Regression** model trained on the provided dataset.

---

## 🚀 Project Steps (Summary)

### Step 2: Model Development (Google Colab)
1. Load dataset (`Housing.csv`)
2. Split into train/test sets
3. Train a **LinearRegression** model
4. Evaluate (MAE, R²)
5. Save as `model.pkl` using `joblib`

### Step 3: Streamlit Implementation (VS Code)
Endpoints:
- `GET /` → Health check
- `POST /predict` → Make a prediction
- `GET /model-info` → Model details (features, metrics)

Features:
- Input validation with **Pydantic**
- Error handling with **HTTPException**
- JSON response format
- Model loaded on startup
