import requests

# 1. SETUP: Define the endpoint URL
url = "https://irctc1.p.rapidapi.com/api/v1/liveTrainStatus"

# 2. INPUT: Enter your API Key and Train Details here
# Replace 'YOUR_RAPIDAPI_KEY_HERE' with the long string found in the "Header Parameters" section of your RapidAPI dashboard.
my_api_key = "f44e7c30-2f29-4686-99a5-84deb9cf1fa7" 

# Train Number: 19038 (Avadh Express)
# startDay: 0 = Today, 1 = Yesterday (This refers to when the train started its journey)
query_params = {
    "trainNo": "19038", 
    "startDay": "1"
}

# 3. HEADERS: This is where the API Key authenticates you
headers = {
    "x-rapidapi-key": my_api_key,
    "x-rapidapi-host": "irctc1.p.rapidapi.com"
}

# 4. REQUEST: Send the data to the API
try:
    response = requests.get(url, headers=headers, params=query_params)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        print("--- API Response ---")
        print(data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")