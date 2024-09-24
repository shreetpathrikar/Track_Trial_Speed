from flask import Flask, jsonify, request, render_template
import googlemaps
import math
import nltk
from nltk import word_tokenize
import torch
from transformers import pipeline
import soundfile as sf
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


app = Flask(__name__)

# Initialize Google Maps client (API key required)
gmaps = googlemaps.Client(key='AIzaSyDP9Cyr-AzJGGXxZORu7Lee7rwhvtYtdTo')

# Hardcoded speed limits for demonstration purposes
speed_limits = {
    "Kanadiya Road": 40,
    "M G Road": 50,
    "Ring Road": 60,
    "Badi Gwaltoli Rd": 45,
    "Bhandari Marg": 30,
    "Mahatma Gandhi Rd": 35,
    "Race Course": 30,
    "Curewell Hospital Rd": 25,
    "AB Rd": 40,
    "Greater Kailash Rd": 40,
    "Unknown": 10  # Default speed limit if not found
}

@app.route('/')
def index():
    return render_template('index.html')





# Route to fetch nearby police stations
@app.route('/get_nearby_police_stations', methods=['GET'])
def get_nearby_police_stations():
    user_lat = request.args.get('lat', type=float)
    user_lng = request.args.get('lng', type=float)

    # Google Maps Places API to find nearby police stations
    places_result = gmaps.places_nearby(
        location=(user_lat, user_lng), 
        radius=5000,  # Search within 5km radius
        type="police"  # Search for police stations
    )
    
    police_stations = []
    for place in places_result['results']:
        station = {
            "name": place['name'],
            "vicinity": place['vicinity'],
            "lat": place['geometry']['location']['lat'],
            "lng": place['geometry']['location']['lng']
        }
        
        # Calculate distance from user location
        distance_result = gmaps.distance_matrix(
            origins=f"{user_lat},{user_lng}",
            destinations=f"{station['lat']},{station['lng']}",
            mode="driving"
        )
        distance = distance_result['rows'][0]['elements'][0]['distance']['text']
        station["distance"] = distance
        
        police_stations.append(station)
    
    # Sort by distance
    sorted_stations = sorted(police_stations, key=lambda k: k['distance'])
    
    # Return nearest station for index.html and top 5 for police.html
    nearest_station = sorted_stations[0] if sorted_stations else None
    top_5_stations = sorted_stations[:5]
    
    return jsonify({"nearest": nearest_station, "top_5": top_5_stations})






# Route to fetch nearby hospitals
@app.route('/get_nearby_hospitals', methods=['GET'])
def get_nearby_hospitals():
    user_lat = request.args.get('lat', type=float)
    user_lng = request.args.get('lng', type=float)

    # Google Maps Places API to find nearby hospitals
    places_result = gmaps.places_nearby(
        location=(user_lat, user_lng), 
        radius=5000,  # Search within 5km radius
        type="hospital"  # Search for hospitals
    )
    
    hospitals = []
    for place in places_result['results']:
        hospital = {
            "name": place['name'],
            "vicinity": place['vicinity'],
            "lat": place['geometry']['location']['lat'],
            "lng": place['geometry']['location']['lng']
        }
        
        # Calculate distance from user location
        distance_result = gmaps.distance_matrix(
            origins=f"{user_lat},{user_lng}",
            destinations=f"{hospital['lat']},{hospital['lng']}",
            mode="driving"
        )
        distance = distance_result['rows'][0]['elements'][0]['distance']['text']
        hospital["distance"] = distance
        
        hospitals.append(hospital)
    
    # Sort by distance
    sorted_hospitals = sorted(hospitals, key=lambda k: k['distance'])
    
    # Return top 5 hospitals
    return jsonify({"hospitals": sorted_hospitals[:5]})








# Route to fetch nearby fuel stations
@app.route('/get_nearby_fuel_stations', methods=['GET'])
def get_nearby_fuel_stations():
    user_lat = request.args.get('lat', type=float)
    user_lng = request.args.get('lng', type=float)

    if user_lat is None or user_lng is None:
        return jsonify({"error": "Missing latitude or longitude parameters."}), 400

    places_result = gmaps.places_nearby(
        location=(user_lat, user_lng), 
        radius=5000,  
        type="gas_station"  
    )

    fuels = []
    for place in places_result.get('results', []):
        fuel = {
            "name": place.get('name'),
            "vicinity": place.get('vicinity'),
            "lat": place['geometry']['location']['lat'],
            "lng": place['geometry']['location']['lng']
        }
        fuels.append(fuel)

    # Debugging: Print the results for inspection
    # print("Fuel Stations Found:", fuels)

    top_fuels = fuels[:5]  # Limit to top 5 fuel stations
    return jsonify({"fuels": top_fuels})

















# Route to get speed limit based on location (road name)
@app.route('/get_speed_limit/<location>', methods=['GET'])
def get_speed_limit(location):
    location = location.lower()  # Make location lowercase for easier matching
    for road_name, speed_limit in speed_limits.items():
        if road_name.lower() in location:
            return jsonify({"speed_limit": speed_limit})
    # Return default speed limit if no match
    return jsonify({"speed_limit": speed_limits["Unknown"]})

# Function to calculate speed based on coordinates and time difference
def calculate_speed(lat1, lon1, lat2, lon2, time_difference):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Distance in meters
    speed_m_per_s = distance / time_difference  # Speed in meters per second
    speed_km_per_h = speed_m_per_s * 3.6  # Convert to km/h
    return speed_km_per_h

# API route to calculate speed based on coordinates
@app.route('/calculate_speed', methods=['POST'])
def get_speed():
    data = request.json
    lat1 = data['lat1']
    lon1 = data['lon1']
    lat2 = data['lat2']
    lon2 = data['lon2']
    time_difference = data['time_difference']

    speed = calculate_speed(lat1, lon1, lat2, lon2, time_difference)
    return jsonify({"speed": speed})
















# Download the necessary NLTK data if you haven't already
nltk.download('punkt')

# Speed limits based on institution type
speed_limit_voice = {
    "school": 20,
    "hospital": 10,
    "mandir": 30
}

# Initialize your text-to-speech model
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

def convert_number_to_words(num):
    """Convert numbers to words using nltk."""
    from num2words import num2words
    return num2words(num)

def generate_speech(message):
    """Generate speech from text and save to a file."""
    speech = synthesiser(message)
    audio_file_path = "static/speech.wav"  # Save in a static folder
    sf.write(audio_file_path, speech["audio"], samplerate=speech["sampling_rate"])
    return audio_file_path

@app.route('/get_nearby_institutions', methods=['GET'])
def get_nearby_institutions():
    user_lat = request.args.get('lat', type=float)
    user_lng = request.args.get('lng', type=float)

    if user_lat is None or user_lng is None:
        return jsonify({"error": "Missing latitude or longitude parameters."}), 400

    places = []
    types = ["school", "hospital", "hindu_temple"]  # Adjust type for mandir

    for place_type in types:
        places_result = gmaps.places_nearby(
            location=(user_lat, user_lng),
            radius=500,
            type=place_type
        )
        for place in places_result.get('results', []):
            places.append({
                "name": place.get('name'),
                "vicinity": place.get('vicinity'),
                "lat": place['geometry']['location']['lat'],
                "lng": place['geometry']['location']['lng'],
                "type": place_type
            })

    # Create a speed limit message and generate speech
    messages = []
    audio_files = []
    for place in places:
        institution_type = place['type']
        speed_limit = speed_limit_voice.get(institution_type, None)
        if speed_limit is not None:
            speed_limit_in_words = convert_number_to_words(speed_limit)
            message = f"Please limit your speed to {speed_limit_in_words} km/h in front of {place['name']}"
            messages.append(message)
            audio_file_path = generate_speech(message)
            audio_files.append(audio_file_path)

    return jsonify({"messages": messages, "audio_files": audio_files})















@app.route('/police.html')
def police():
    return render_template('police.html')


@app.route('/hospital.html')
def hospital():
    return render_template('hospital.html')


@app.route('/fuel.html')
def fuel():
    return render_template('/fuel.html')


if __name__ == '__main__':
    app.run(debug=True)
