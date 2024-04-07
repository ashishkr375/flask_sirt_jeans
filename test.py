import requests

# Example data for testing
data = {
    'City': 'New York',
    'Temperature': 25  # Assume temperature in Celsius
}

# URL of the Flask app
url = 'http://localhost:5000/recommend-food'  # Change to your actual URL

# Send POST request to the Flask app
response = requests.post(url, json=data)

# Print the response
print(response.json())
