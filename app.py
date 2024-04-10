from flask import Flask, request, jsonify
from flask_cors import CORS 
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import pickle
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  


# Load the trained model for shirt and jeans prediction
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Configure Gemini AI
genai.configure(api_key='AIzaSyDvgQy57uiGhlZrWA6NZJArR__P_IwBA18')

# Function to get food recommendations using Gemini AI
def get_food_recommendation(city, temperature):
    input_prompt = f"""
        You are an expert in food recommendations.
        You have been asked to suggest meals for a day based on the weather in {city} and temperature of {temperature}°C.
        
        Please include suitable foods considering the current weather conditions.
        like if it's summer, please include refreshing and hydrating foods in the meal plan, such as:
            - Salads with seasonal vegetables and fruits
            - Grilled meats or fish
            - Cold soups (e.g., gazpacho)
            - Smoothies or fruit juices
            - Light and refreshing desserts (e.g., sorbet, fruit salad)
        else if it's winter, please include warming and comforting foods in the meal plan, such as:
            - Hearty soups (e.g., chicken noodle soup, minestrone)
            - Stews and casseroles (e.g., beef stew, shepherd's pie)
            - Roasted vegetables (e.g., carrots, potatoes, Brussels sprouts)
            - Hot beverages like tea, coffee, or hot chocolate
            - Oatmeal or porridge for breakfast
        importantly provide a balanced meal plan for the entire day, including breakfast, lunch, snacks, and dinner.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_prompt])
    return response.text

# Endpoint for making predictions for shirt and jeans
@app.route('/predict', methods=['POST'])
def predict_clothes():
    try:
        # Get the JSON data from the request
        data = request.json

        # Extract features from the JSON data for shirt and jeans prediction
        features = [data['Temperature'], data['Humidity'], data['Wind speed']]

        # Make prediction using the loaded model for shirt and jeans
        predictions = model.predict([features])

        # Separate the predictions for 'Shirt' and 'Jeans'
        predicted_shirt = predictions[:, 0][0]  # Get the first prediction for 'Shirt'
        predicted_jeans = predictions[:, 1][0]  # Get the first prediction for 'Jeans'

        # Prepare the response JSON
        response = {
            'Shirt': predicted_shirt,
            'Jeans': predicted_jeans
        }

        # Return the response as JSON
        return jsonify(response), 200

    except Exception as e:
        # Return error response if there's an exception
        error_message = {'error': str(e)}
        return jsonify(error_message), 500

# Endpoint for getting food recommendation
@app.route('/recommend-food', methods=['POST'])
def recommend_food():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Get food recommendation based on temperature and city
        food_recommendation = get_food_recommendation(data['City'], data['Temperature'])

        # Prepare the response JSON
        response = {
            'FoodRecommendation': food_recommendation
        }

        # Return the response as JSON
        return jsonify(response), 200

    except Exception as e:
        # Return error response if there's an exception
        error_message = {'error': str(e)}
        return jsonify(error_message), 500

# Function to get activity suggestions
def get_activity_suggestion(city, temperature, weather_type, wind_speed, humidity):
    input_prompt = f"""
        You are an expert in suggesting activities.
        You have been asked to recommend activities based on the following weather conditions in {city}:
        - Temperature: {temperature}°C
        - Weather: {weather_type}
        - Wind Speed: {wind_speed} Km/s
        - Humidity: {humidity}%
        
        Please suggest both indoor and outdoor activities suitable for these conditions.
        For example, for sunny weather, suggest outdoor activities like:
            - Picnic in the park
            - Hiking
            - Cycling
            - Beach outing
        For rainy weather, suggest indoor activities like:
            - Movie marathon
            - Board games
            - Cooking or baking
            - Indoor rock climbing
        Make sure to consider the safety and comfort of the individuals while suggesting activities.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_prompt])
    return response.text

@app.route('/recommend-activity', methods=['POST'])
def recommend_activity():
    try:
        data = request.json
        city = data['City']
        temperature = data['Temperature']
        weather_type = data['WeatherType']
        wind_speed = data['Wind_speed']
        humidity = data['Humidity']
        
        # Get activity recommendation based on weather conditions
        activity_suggestion = get_activity_suggestion(city, temperature, weather_type, wind_speed, humidity)

        # Prepare the response JSON
        response = {
            'ActivitySuggestion': activity_suggestion
        }

        # Return the response as JSON
        return jsonify(response), 200

    except Exception as e:
        # Return error response if there's an exception
        error_message = {'error': str(e)}
        return jsonify(error_message), 500
def get_place_suggestion(city, temperature, weather_type, wind_speed, humidity):
    input_prompt = f"""
        You are an expert in suggesting activities.
        You have been asked to recommend activities based on the following weather conditions in {city}:
        - Temperature: {temperature}°C
        - Weather: {weather_type}
        - Wind Speed: {wind_speed} Km/s
        - Humidity: {humidity}%
        
        Please suggest both indoor and outdoor places suitable for these conditions.
        
        For open places:
        - Recommend nearby parks, hiking trails, beaches, or outdoor attractions in {city}.
        - Provide Google Maps links to these locations if available.
        
        For close places:
        - Suggest indoor entertainment options like museums, cafes, theaters, or indoor sports facilities in {city}.
        - Provide Google Maps links to these locations if available.
        
        Additionally, suggest some outstation destinations suitable for the current weather conditions.
        - Recommend popular nearby tourist spots or day trip destinations.
        - Provide Google Maps links to these locations if available.
        
        Make sure to consider the safety and comfort of the individuals while suggesting activities.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_prompt])
    return response.text


@app.route('/places-to-visit', methods=['POST'])
def recommend_place():
    try:
        data = request.json
        city = data['City']
        temperature = data['Temperature']
        weather_type = data['WeatherType']
        wind_speed = data['Wind_speed']
        humidity = data['Humidity']
        
        # Get activity recommendation based on weather conditions
        activity_suggestion_place = get_place_suggestion(city, temperature, weather_type, wind_speed, humidity)

        # Prepare the response JSON
        response = {
            'PlaceSuggestion': activity_suggestion_place
        }

        # Return the response as JSON
        return jsonify(response), 200

    except Exception as e:
        # Return error response if there's an exception
        error_message = {'error': str(e)}
        return jsonify(error_message), 500


if __name__ == '__main__':
    app.run(debug=True)
