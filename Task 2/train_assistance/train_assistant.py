import streamlit as st
import google.generativeai as genai
import requests
import json
import sqlite3
from datetime import datetime, timedelta
import re

# Configure Gemini API
GEMINI_API_KEY = "USE API KEY"
genai.configure(api_key=GEMINI_API_KEY)

# RailRadar API Base URL
RAILRADAR_API_BASE = "https://railradar.in/api/v1"

# Initialize SQLite Database
def init_database():
    conn = sqlite3.connect('train_queries.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  user_query TEXT,
                  api_calls TEXT,
                  response TEXT)''')
    conn.commit()
    conn.close()

def save_query(user_query, api_calls, response):
    conn = sqlite3.connect('train_queries.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO queries (timestamp, user_query, api_calls, response) VALUES (?, ?, ?, ?)",
              (timestamp, user_query, json.dumps(api_calls), response))
    conn.commit()
    conn.close()

# RailRadar API Functions
def search_stations(query):
    """Search for stations by code or name"""
    try:
        response = requests.get(f"{RAILRADAR_API_BASE}/search/stations", 
                              params={"q": query})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_live_station_board(station_code, hours=8, to_station_code=None):
    """Get live station board with departures/arrivals"""
    try:
        params = {"hours": hours}
        if to_station_code:
            params["toStationCode"] = to_station_code
        
        response = requests.get(
            f"{RAILRADAR_API_BASE}/stations/{station_code}/live",
            params=params
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_trains_between_stations(from_code, to_code):
    """Get trains between two stations"""
    try:
        response = requests.get(
            f"{RAILRADAR_API_BASE}/trains/between",
            params={"from": from_code, "to": to_code}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_train_live_status(train_number, journey_date=None):
    """Get live status of a train"""
    try:
        params = {"dataType": "live"}
        if journey_date:
            params["journeyDate"] = journey_date
        
        response = requests.get(
            f"{RAILRADAR_API_BASE}/trains/{train_number}",
            params=params
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def search_trains(query):
    """Search for trains by number or name"""
    try:
        response = requests.get(f"{RAILRADAR_API_BASE}/search/trains",
                              params={"q": query})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

# LLM Function Calling Setup
tools = [
    {
        "name": "search_stations",
        "description": "Search for railway stations by code or name. Use this to find station codes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Station name or code to search for (e.g., 'Nahur', 'CSMT', 'Mumbai')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_live_station_board",
        "description": "Get live departure and arrival information for a station. Shows trains leaving/arriving at a station with real-time delays.",
        "parameters": {
            "type": "object",
            "properties": {
                "station_code": {
                    "type": "string",
                    "description": "Station code (e.g., 'NR' for Nahur, 'CSMT' for Chhatrapati Shivaji Maharaj Terminus)"
                },
                "hours": {
                    "type": "integer",
                    "description": "Number of hours to look ahead (1-8, default 8)"
                },
                "to_station_code": {
                    "type": "string",
                    "description": "Optional: Filter trains going to this destination station code"
                }
            },
            "required": ["station_code"]
        }
    },
    {
        "name": "get_trains_between_stations",
        "description": "Find all trains that run between two stations. Use this for journey planning.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_code": {
                    "type": "string",
                    "description": "Source station code"
                },
                "to_code": {
                    "type": "string",
                    "description": "Destination station code"
                }
            },
            "required": ["from_code", "to_code"]
        }
    },
    {
        "name": "get_train_live_status",
        "description": "Get real-time running status of a specific train including current location, delays, and estimated arrival times.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_number": {
                    "type": "string",
                    "description": "5-digit train number (e.g., '12345')"
                },
                "journey_date": {
                    "type": "string",
                    "description": "Optional: Journey date in YYYY-MM-DD format"
                }
            },
            "required": ["train_number"]
        }
    },
    {
        "name": "search_trains",
        "description": "Search for trains by train number or name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Train number or name to search for"
                }
            },
            "required": ["query"]
        }
    }
]

# Function execution mapper
function_map = {
    "search_stations": search_stations,
    "get_live_station_board": get_live_station_board,
    "get_trains_between_stations": get_trains_between_stations,
    "get_train_live_status": get_train_live_status,
    "search_trains": search_trains
}

def process_query_with_llm(user_query):
    """Process user query using Gemini with function calling"""
    
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash-exp',
        tools=tools
    )
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = f"""You are a helpful Indian Railways assistant. Current date and time: {current_time}

When users ask about trains:
1. First search for station codes if you don't have them
2. Then use appropriate APIs to get live information
3. Always provide real-time delays and actual timings when available
4. Format your response in a clear, user-friendly manner
5. If asking about "next train", use the live station board to show upcoming departures

Common station codes:
- CSMT: Chhatrapati Shivaji Maharaj Terminus
- NR: Nahur
- DR: Dadar
- KYN: Kalyan
- TNA: Thane
- BVI: Borivali
"""
    
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    api_calls_made = []
    
    try:
        # Send the query
        response = chat.send_message(system_prompt + "\n\nUser query: " + user_query)
        
        # Track function calls
        for content in chat.history:
            if hasattr(content, 'parts'):
                for part in content.parts:
                    if hasattr(part, 'function_call'):
                        api_calls_made.append({
                            'function': part.function_call.name,
                            'args': dict(part.function_call.args)
                        })
        
        return response.text, api_calls_made
    
    except Exception as e:
        return f"Error processing query: {str(e)}", api_calls_made

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Indian Railways Assistant",
        page_icon="🚆",
        layout="wide"
    )
    
    # Initialize database
    init_database()
    
    # Header
    st.title("🚆 Indian Railways Live Information Assistant")
    st.markdown("*Ask me anything about trains, stations, and live running status!*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Features")
        st.markdown("""
        - 🔍 Search trains and stations
        - 🚉 Live station boards
        - ⏱️ Real-time train status
        - 🛤️ Trains between stations
        - 📝 Query history
        """)
        
        st.divider()
        
        st.header("💡 Example Queries")
        st.markdown("""
        - "When is the next train from Nahur to CSMT?"
        - "Show me live status of train 12345"
        - "Which trains run between Mumbai and Pune?"
        - "What's the delay on Rajdhani Express?"
        - "Show departures from Dadar station"
        """)
        
        st.divider()
        
        if st.button("📜 View Query History"):
            st.session_state.show_history = True
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "api_calls" in message and message["api_calls"]:
                with st.expander("🔧 API Calls Made"):
                    st.json(message["api_calls"])
    
    # Chat input
    if prompt := st.chat_input("Ask about trains, stations, or live status..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with LLM
        with st.chat_message("assistant"):
            with st.spinner("Fetching live information..."):
                response, api_calls = process_query_with_llm(prompt)
                st.markdown(response)
                
                if api_calls:
                    with st.expander("🔧 API Calls Made"):
                        st.json(api_calls)
                
                # Save to database
                save_query(prompt, api_calls, response)
                
                # Add to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "api_calls": api_calls
                })
    
    # Query History Modal
    if st.session_state.get("show_history", False):
        st.divider()
        st.header("📜 Query History")
        
        conn = sqlite3.connect('train_queries.db')
        c = conn.cursor()
        c.execute("SELECT * FROM queries ORDER BY id DESC LIMIT 20")
        queries = c.fetchall()
        conn.close()
        
        if queries:
            for query in queries:
                with st.expander(f"🕐 {query[1]} - {query[2][:50]}..."):
                    st.markdown(f"**Query:** {query[2]}")
                    st.markdown(f"**Response:** {query[4]}")
                    if query[3]:
                        st.json(json.loads(query[3]))
        else:
            st.info("No query history yet!")
        
        if st.button("Close History"):
            st.session_state.show_history = False
            st.rerun()

if __name__ == "__main__":
    main()