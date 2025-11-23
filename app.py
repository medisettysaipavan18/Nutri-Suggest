

# app.py
import os
import re
import difflib
import json
import logging
import cv2
import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

# ---------- Flask app config ----------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Load model/encoder/dataset ----------
model = tf.keras.models.load_model("food_cnn_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
try:
    model = tf.keras.models.load_model("food_cnn_model.h5")
    label_encoder = joblib.load("label_encoder.pkl")
    data = pd.read_csv(r'nutrition.csv')
    logger.info("Loaded model, label encoder, and dataset.")
except Exception as e:
    logger.warning(f"Error loading model or data files: {e}")
    model = None
    label_encoder = None
    data = pd.DataFrame({
        'name': ['apple'],
        'calories': [95],
        'protein': [0.5],
        'total_fat': [0.3],
        'serving_size': ['1 medium']
    })

# ---------- Helper Functions ----------
def convert_measurement(value):
    try:
        value = str(value)
        value = re.sub(r'\s+', '', value)  # Remove extra spaces
        if value == '':
            return np.nan
        if 'mcg' in value:
            return float(value.replace("mcg", "")) / 1000000.0
        elif 'mg' in value:
            return float(value.replace("mg", "")) / 1000.0
        elif value.endswith('g') and not value.endswith('mg') and not value.endswith('mcg'):
            return float(value.replace("g", ""))
        elif 'IU' in value:
            return float(value.replace("IU", "")) * 0.025 / 1000000.0
        else:
            return float(value)
    except (ValueError, TypeError):
        return np.nan

if not data.empty:
    columns_to_convert = [c for c in data.columns if c not in ['name', 'serving_size']]
    for column in columns_to_convert:
        try:
            data[column] = data[column].apply(convert_measurement)
        except Exception:
            pass
    data = data.dropna(subset=['name'], how='any')

def closest_match(dish_name):
    if data.empty or 'name' not in data.columns:
        return None
    highest_ratio = 0
    best_match = None
    for name in data['name']:
        try:
            similarity_ratio = difflib.SequenceMatcher(None, str(dish_name).lower(), str(name).lower()).ratio()
            if similarity_ratio > highest_ratio:
                highest_ratio = similarity_ratio
                best_match = name
        except Exception:
            continue
    return best_match

def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def calculate_food_area_and_size(image):
    image_np = np.array(image)
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        food_area = sum(cv2.contourArea(contour) for contour in contours)
    except Exception:
        mask = np.any(image_np < 250, axis=2)
        food_area = int(mask.sum())

    if food_area < 10000:
        size = 'small'
    elif food_area < 30000:
        size = 'medium'
    else:
        size = 'large'
    return food_area, size

def predict_food_nutrition(dish_name, food_area):
    closest_dish = closest_match(dish_name)
    if closest_dish and not data.empty:
        dish_data = data[data['name'] == closest_dish]
        if not dish_data.empty:
            factor = float(food_area) / 300000.0 if food_area > 0 else 1.0
            try:
                calories = float(dish_data['calories'].values[0]) * factor
                protein = float(dish_data['protein'].values[0]) * factor
                total_fat = float(dish_data['total_fat'].values[0]) * factor
                return {
                    "calories": f"{calories:.2f} kcal",
                    "protein": f"{protein:.2f} g",
                    "total_fat": f"{total_fat:.2f} g",
                    "matched_name": closest_dish
                }
            except Exception:
                return {"error": "Nutrition data invalid for matched dish."}
    return {"error": "Dish not found in the dataset."}

# ---------- Gemini (Generative Language) helpers ----------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

def list_gemini_models(api_key):
    url = f"{GEMINI_BASE}/models?key={api_key}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def _clean_model_name(raw_name: str) -> str:
    """
    Normalize a model name so it does NOT contain a leading 'models/'.
    Examples:
      'models/gemini-1.5-flash-latest' -> 'gemini-1.5-flash-latest'
      'gemini-1.5-flash-latest' -> 'gemini-1.5-flash-latest'
    """
    if not raw_name:
        return raw_name
    # If the API returned a full resource name like "models/gemini-..."
    if raw_name.startswith("models/"):
        return raw_name.split("/", 1)[1]
    # Sometimes entries include the full path: "projects/PROJECT/locations/global/models/gemini-..."
    if "/models/" in raw_name:
        return raw_name.split("/models/")[-1]
    return raw_name

def choose_model_from_list(models_json):
    """
    Extract and return a cleaned model name (without leading 'models/').
    Preference order: gemini-*flash*, then any gemini, else first available.
    """
    model_entries = models_json.get('models') if isinstance(models_json, dict) else None
    if not model_entries:
        return None
    candidates = []
    for m in model_entries:
        # API may have 'name' or 'model'
        raw = m.get('name') or m.get('model') or ''
        cleaned = _clean_model_name(raw)
        if cleaned:
            candidates.append(cleaned)
    # prefer flash
    for name in candidates:
        if 'flash' in name.lower() and 'gemini' in name.lower():
            return name
    for name in candidates:
        if 'gemini' in name.lower():
            return name
    return candidates[0] if candidates else None

def parse_generate_response(resp_json):
    if not isinstance(resp_json, dict):
        return str(resp_json)
    try:
        if 'candidates' in resp_json:
            c = resp_json['candidates']
            if isinstance(c, list) and len(c) > 0:
                try:
                    return c[0]['content']['parts'][0]['text']
                except Exception:
                    pass
        if 'outputs' in resp_json:
            outputs = resp_json['outputs']
            if isinstance(outputs, list) and len(outputs) > 0:
                try:
                    return outputs[0]['content'][0]['text']
                except Exception:
                    pass
        def find_text(d):
            if isinstance(d, dict):
                for k,v in d.items():
                    if k == 'text' and isinstance(v, str):
                        return v
                    r = find_text(v)
                    if r: return r
            elif isinstance(d, list):
                for item in d:
                    r = find_text(item)
                    if r: return r
            return None
        text = find_text(resp_json)
        if text:
            return text
    except Exception:
        pass
    return json.dumps(resp_json, indent=2, ensure_ascii=False)

def call_gemini_api(food_name, nutrition, user_query, api_key=None, explicit_model=None):
    api_key = "AIzaSyAbOcThnd7scVhalPWTx0nIWSqN7sah9ps"#api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return "No API key provided. Set GEMINI_API_KEY environment variable on the server."

    try:
        model_name = explicit_model
        if model_name:
            model_name = _clean_model_name(model_name)

        if model_name is None:
            try:
                models_json = list_gemini_models(api_key)
                logger.info(f"ListModels returned: {models_json}")
                model_name = choose_model_from_list(models_json)
                if model_name:
                    logger.info(f"Auto-selected model: {model_name}")
                else:
                    return "No usable Gemini model found for this API key (ListModels returned no usable names)."
            except requests.HTTPError as e:
                raw = e.response.text if e.response is not None else None
                logger.exception("HTTP error listing models")
                return f"Error listing models: {e}. Raw response: {raw}"
            except Exception as e:
                logger.exception("Unexpected error listing models")
                return f"Unexpected error while listing models: {e}"

        prompt = f"Food: {food_name}\nNutrition: {nutrition}\nUser Query: {user_query}\n\nAct As a health expert, answer if this food is safe based on the  user given their query, and if so, recommend a safe quantity. Be concise. , dot ask for any forther details only give the response on the data you get"

        # model_name must NOT include 'models/' prefix
        model_name = _clean_model_name(model_name)
        url = f"{GEMINI_BASE}/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        resp_json = resp.json()
        return parse_generate_response(resp_json)

    except requests.HTTPError as e:
        raw = None
        try:
            raw = e.response.text
        except Exception:
            raw = None
        logger.exception("HTTPError calling Gemini API")
        return f"Error contacting Gemini API: {e}. Raw response: {raw}"
    except Exception as e:
        logger.exception("Unexpected error calling Gemini API")
        return f"Unexpected error contacting Gemini API: {e}"

# ---------- Flask Routes ----------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and model is not None and label_encoder is not None:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                image = Image.open(filepath).convert('RGB')
            except Exception as e:
                return render_template('index.html', error=f'Unable to open image: {e}')

            processed_image = preprocess_image(image)
            try:
                prediction = model.predict(processed_image)
                predicted_class_numeric = int(np.argmax(prediction, axis=1)[0])
                predicted_class = label_encoder.inverse_transform([predicted_class_numeric])[0]
                confidence = float(np.max(prediction))
            except Exception as e:
                logger.exception("Prediction failed")
                return render_template('index.html', error=f'Model prediction failed: {e}')

            food_area, size = calculate_food_area_and_size(image)
            nutrition_info = predict_food_nutrition(predicted_class, food_area)

            return render_template('result.html',
                                   predicted_class=predicted_class,
                                   confidence=f"{confidence:.2f}",
                                   nutrition_info=nutrition_info,
                                   image_filename=filename)
        else:
            return render_template('index.html', error='Model or label encoder not loaded correctly. Check server logs.')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/llm_advice', methods=['POST'])
def llm_advice():
    predicted_class = request.form.get('predicted_class')
    nutrition_calories = request.form.get('nutrition_calories')
    nutrition_protein = request.form.get('nutrition_protein')
    nutrition_fat = request.form.get('nutrition_fat')
    user_query = request.form.get('user_query')
    image_filename = request.form.get('image_filename')

    explicit_model = request.form.get('explicit_model') or None

    nutrition = f"Calories: {nutrition_calories}, Protein: {nutrition_protein}, Total Fat: {nutrition_fat}"

    llm_advice_text = call_gemini_api(predicted_class, nutrition, user_query, api_key=None, explicit_model=explicit_model)

    nutrition_info = {"calories": nutrition_calories, "protein": nutrition_protein, "total_fat": nutrition_fat}

    return render_template('result.html',
                           predicted_class=predicted_class,
                           confidence=None,
                           nutrition_info=nutrition_info,
                           image_filename=image_filename,
                           llm_advice=llm_advice_text)

# ---------- Main ----------
if __name__ == '__main__':
    if not os.environ.get('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY not set. Set it as an environment variable before calling the LLM endpoints.")
    app.run(debug=True, host='0.0.0.0', port=5000)
