# AI-Based Food Nutrition Detection & Diet Suggestion System #

A web application powered by Deep Learning and Generative AI to recognize food, estimate portion size, calculate nutrition, and provide personalized diet advice.

---

## Overview

The system combines Computer Vision (CNNs), Image Processing (OpenCV), and Generative AI (Google Gemini) to deliver:

- Food recognition from uploaded images
- Portion size estimation (small/medium/large)
- Calorie & macronutrient calculation
- Personalized diet recommendations

Workflow:
1. Upload food image
2. CNN predicts food type
3. Portion size estimated using OpenCV
4. Nutrition calculated
5. Gemini API generates diet advice

---

## Features

- Automatic food recognition using CNN
- Portion size estimation via OpenCV
- Nutrient calculation (calories, protein, fat)
- Personalized diet suggestions via Gemini API
- Flask-based web interface
- Error handling and fallback logic

---

## Tech Stack

| Component            | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Deep Learning        | TensorFlow / Keras              |
| Image Processing     | OpenCV, PIL                     |
| Web Framework        | Flask                           |
| Data Handling        | Pandas, NumPy                   |
| LLM Integration      | Google Gemini API               |
| Model Storage        | Joblib                          |
| Frontend             | HTML, CSS (Flask templates)     |

---
