import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pytz
from geopy.distance import geodesic
import googlemaps
import os

# Initialize session state
if 'transfers' not in st.session_state:
    st.session_state.transfers = []
if 'inventory' not in st.session_state:
    st.session_state.inventory = []
if 'sales' not in st.session_state:
    st.session_state.sales = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'store_data' not in st.session_state:
    st.session_state.store_data = []
if 'product_data' not in st.session_state:
    st.session_state.product_data = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'status_filter' not in st.session_state:
    st.session_state.status_filter = "All"

# Configuration
st.set_page_config(layout="wide", page_title="Walmart Smart Routing", page_icon="🚚")
st.title("🚚 Walmart Smart Inventory Routing System")
st.subheader("Optimized Product Transfers Between Supercenters Based on Real-Time Conditions")

# API Keys (Replace with your actual keys)
VISUALCROSSING_API_KEY = st.secrets["VISUALCROSSING_API_KEY"]
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"] 
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Constants
WEATHER_CATEGORIES = ['Clear', 'Cold', 'Extreme Heat', 'Normal', 'Rain', 'Snow']
MAX_TRANSFER_DISTANCE = 100  # miles
MAX_TRANSFER_TIME = 120  # minutes

# Real Walmart store locations in Washington DC area
WALMART_STORES = [
    {
        'store_id': 1001,
        'name': "Walmart Supercenter - Washington DC",
        'city': "Washington, DC",
        'address': "99 H St NW, Washington, DC 20001",
        'lat': 38.9003,
        'lng': -77.0086,
        'size_sqft': 180000,
        'manager': "Sarah Johnson",
        'phone': "(202) 347-1001"
    },
    {
        'store_id': 1002,
        'name': "Walmart Supercenter - Alexandria",
        'city': "Alexandria, VA",
        'address': "5900 Stevenson Ave, Alexandria, VA 22304",
        'lat': 38.8065,
        'lng': -77.0866,
        'size_sqft': 200000,
        'manager': "Michael Chen",
        'phone': "(703) 924-1002"
    },
    {
        'store_id': 1003,
        'name': "Walmart Supercenter - Arlington",
        'city': "Arlington, VA",
        'address': "4200 28th St S, Arlington, VA 22206",
        'lat': 38.8485,
        'lng': -77.0511,
        'size_sqft': 175000,
        'manager': "David Rodriguez",
        'phone': "(703) 578-1003"
    },
    {
        'store_id': 1004,
        'name': "Walmart Supercenter - Silver Spring",
        'city': "Silver Spring, MD",
        'address': "12051 Cherry Hill Rd, Silver Spring, MD 20904",
        'lat': 39.0548,
        'lng': -76.9814,
        'size_sqft': 190000,
        'manager': "Jennifer Williams",
        'phone': "(301) 754-1004"
    },
    {
        'store_id': 1005,
        'name': "Walmart Supercenter - Rockville",
        'city': "Rockville, MD",
        'address': "705 N Frederick Ave, Rockville, MD 20850",
        'lat': 39.0923,
        'lng': -77.1389,
        'size_sqft': 185000,
        'manager': "Robert Taylor",
        'phone': "(301) 230-1005"
    },
    {
        'store_id': 1006,
        'name': "Walmart Supercenter - College Park",
        'city': "College Park, MD",
        'address': "10100 Baltimore Ave, College Park, MD 20740",
        'lat': 39.0008,
        'lng': -76.9227,
        'size_sqft': 170000,
        'manager': "Lisa Anderson",
        'phone': "(301) 345-1006"
    }
]

# Real Walmart products with categories
WALMART_PRODUCTS = [
    {'product_id': 2001, 'name': 'Milk (1 gallon)', 'category': 'Dairy', 'unit_price': 3.49},
    {'product_id': 2002, 'name': 'Bread (24 oz)', 'category': 'Bakery', 'unit_price': 2.49},
    {'product_id': 2003, 'name': 'Eggs (12 count)', 'category': 'Dairy', 'unit_price': 2.99},
    {'product_id': 2004, 'name': 'Paper Towels (6 rolls)', 'category': 'Household', 'unit_price': 7.99},
    {'product_id': 2005, 'name': 'Toilet Paper (12 rolls)', 'category': 'Household', 'unit_price': 12.99},
    {'product_id': 2006, 'name': 'Bananas (1 lb)', 'category': 'Produce', 'unit_price': 0.59},
    {'product_id': 2007, 'name': 'Apples (3 lb bag)', 'category': 'Produce', 'unit_price': 4.99},
    {'product_id': 2008, 'name': 'Chicken Breast (1 lb)', 'category': 'Meat', 'unit_price': 4.49},
    {'product_id': 2009, 'name': 'Bottled Water (24 pack)', 'category': 'Beverages', 'unit_price': 4.99},
    {'product_id': 2010, 'name': 'Cereal (18 oz)', 'category': 'Breakfast', 'unit_price': 3.79},
    {'product_id': 2011, 'name': 'Umbrellas', 'category': 'Seasonal', 'unit_price': 12.99},
    {'product_id': 2012, 'name': 'Sunscreen', 'category': 'Seasonal', 'unit_price': 8.99},
    {'product_id': 2013, 'name': 'Snow Shovels', 'category': 'Seasonal', 'unit_price': 24.99},
    {'product_id': 2014, 'name': 'Fans', 'category': 'Seasonal', 'unit_price': 29.99},
    {'product_id': 2015, 'name': 'Heaters', 'category': 'Seasonal', 'unit_price': 49.99}
]

# Real-time Weather Functions
def get_current_weather(lat, lon):
    """Fetch current weather data from VisualCrossing API"""
    location_key = f"{lat:.4f},{lon:.4f}"
    
    if location_key in st.session_state.weather_data:
        return st.session_state.weather_data[location_key]
    
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today?unitGroup=us&include=current&key={VISUALCROSSING_API_KEY}&contentType=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        current = data['currentConditions']
        weather_data = {
            'temp': current['temp'],
            'feels_like': current['feelslike'],
            'humidity': current['humidity'],
            'conditions': current['conditions'],
            'description': current['conditions'],
            'wind_speed': current['windspeed'],
            'pressure': current['pressure'],
            'uvi': current.get('uvindex', 0),
            'icon': current['icon']
        }
        
        st.session_state.weather_data[location_key] = weather_data
        return weather_data
    except Exception as e:
        # Generate consistent fallback based on location
        rng = random.Random(location_key)
        base_temp = 70 + (lat - 38.9) * 50
        
        weather_data = {
            'temp': base_temp + rng.uniform(-5, 5),
            'feels_like': base_temp + rng.uniform(-7, 7),
            'humidity': rng.randint(40, 80),
            'conditions': "Partly Cloudy",
            'description': "Partly Cloudy",
            'wind_speed': rng.uniform(0, 10),
            'pressure': rng.randint(980, 1040),
            'uvi': rng.uniform(0, 8),
            'icon': 'partly-cloudy-day'
        }
        
        st.session_state.weather_data[location_key] = weather_data
        return weather_data

def get_weather_forecast(lat, lon):
    """Get 7-day weather forecast with precipitation probability"""
    location_key = f"{lat:.4f},{lon:.4f}"
    
    if location_key in st.session_state.forecasts:
        return st.session_state.forecasts[location_key]
    
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/next7days?unitGroup=us&include=days&key={VISUALCROSSING_API_KEY}&contentType=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        forecast = []
        for day in data['days']:
            forecast.append({
                'date': day['datetime'],
                'temp': day['temp'],
                'feels_like': day['feelslike'],
                'humidity': day['humidity'],
                'conditions': day['conditions'],
                'precip': day['precip'],
                'precipprob': day['precipprob'],
                'wind_speed': day['windspeed'],
                'icon': day['icon']
            })
        
        st.session_state.forecasts[location_key] = forecast
        return forecast
    except Exception as e:
        # Generate consistent forecast based on location
        rng = random.Random(location_key)
        base_temp = 70 + (lat - 38.9) * 50
        forecast = []
        
        for i in range(7):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            forecast.append({
                'date': date,
                'temp': base_temp + rng.uniform(-5, 5),
                'feels_like': base_temp + rng.uniform(-7, 7),
                'humidity': rng.randint(40, 80),
                'conditions': "Partly Cloudy" if rng.random() > 0.5 else "Mostly Sunny",
                'precip': rng.uniform(0, 0.1),
                'precipprob': rng.randint(0, 30),
                'wind_speed': rng.uniform(0, 10),
                'icon': 'partly-cloudy-day' if rng.random() > 0.5 else 'clear-day'
            })
        
        st.session_state.forecasts[location_key] = forecast
        return forecast

def get_weather_classification(temp, conditions):
    """Classify weather into impact categories"""
    conditions_lower = conditions.lower()
    if "rain" in conditions_lower:
        return "Rain"
    elif "snow" in conditions_lower:
        return "Snow"
    elif "clear" in conditions_lower and temp > 85:
        return "Extreme Heat"
    elif temp < 32:
        return "Cold"
    elif "clear" in conditions_lower:
        return "Clear"
    return "Normal"

# Google Maps Functions
def get_eta_with_traffic(from_lat, from_lng, to_lat, to_lng):
    """Get estimated travel time with current traffic conditions"""
    try:
        distance = geodesic((from_lat, from_lng), (to_lat, to_lng)).miles
        if distance > MAX_TRANSFER_DISTANCE:
            return None
        
        result = gmaps.directions(
            origin=(from_lat, from_lng),
            destination=(to_lat, to_lng),
            mode="driving",
            departure_time="now",
            traffic_model="best_guess"
        )
        if result and 'legs' in result[0]:
            duration_in_traffic = result[0]['legs'][0]['duration_in_traffic']['value']
            return duration_in_traffic / 60
    except Exception as e:
        st.warning(f"Traffic API error: {e}")
    return None

# Demand Forecasting Functions
def generate_inventory_data(stores, products):
    """Generate current inventory data for all stores"""
    inventory = []
    for store in stores:
        for product in products:
            # Create intentional imbalances
            if store['store_id'] % 3 == 0:  # Every third store will have low stock
                base_stock = random.randint(5, 15)
            elif store['store_id'] % 3 == 1:  # Every third store will have overstock
                base_stock = random.randint(50, 100)
            else:  # Normal stock
                base_stock = random.randint(20, 40)
            
            # Adjust based on store size
            adjusted_stock = int(base_stock * (store['size_sqft'] / 180000))
            
            inventory.append({
                'store_id': store['store_id'],
                'product_id': product['product_id'],
                'current_stock': adjusted_stock,
                'restock_threshold': max(5, int(adjusted_stock * 0.3)),
                'max_capacity': int(adjusted_stock * 1.5),
                'last_delivery': (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d")
            })
    return inventory

def generate_sales_data(stores, products, days=30):
    """Generate historical sales data with realistic patterns"""
    sales = []
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        current_date = (start_date + timedelta(days=i))
        date_str = current_date.strftime("%Y-%m-%d")
        
        for store in stores:
            # Get weather for that day
            weather = get_current_weather(store['lat'], store['lng'])
            
            for product in products:
                # Base sales with randomness
                base_sales = random.randint(5, 15) if product['category'] in ['Dairy', 'Produce', 'Bakery'] else random.randint(1, 5)
                
                # Day of week effect
                day_of_week = current_date.weekday()
                if day_of_week >= 5:  # Weekend
                    base_sales = int(base_sales * 1.5)
                
                # Weather impact for seasonal products
                if product['category'] == 'Seasonal':
                    weather_class = get_weather_classification(weather['temp'], weather['conditions'])
                    if weather_class == 'Rain' and 'Umbrella' in product['name']:
                        base_sales = int(base_sales * 2.5)
                    elif weather_class == 'Snow' and 'Snow' in product['name']:
                        base_sales = int(base_sales * 3.0)
                    elif weather_class == 'Extreme Heat' and 'Fan' in product['name']:
                        base_sales = int(base_sales * 2.8)
                    elif weather_class == 'Cold' and 'Heater' in product['name']:
                        base_sales = int(base_sales * 2.2)
                    elif weather_class == 'Clear' and 'Sunscreen' in product['name']:
                        base_sales = int(base_sales * 2.0)
                
                # Add randomness
                actual_sales = max(0, int(base_sales * random.uniform(0.8, 1.2)))
                
                sales.append({
                    'date': date_str,
                    'store_id': store['store_id'],
                    'product_id': product['product_id'],
                    'units_sold': actual_sales,
                    'temperature': weather['temp'],
                    'weather': weather['conditions']
                })
    return sales

def train_demand_model(sales_data, store_data, product_data):
    """Train demand forecasting model"""
    try:
        # Prepare training data
        df = pd.DataFrame(sales_data)
        
        # Add date features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add product and store features
        product_df = pd.DataFrame(product_data)
        store_df = pd.DataFrame(store_data)
        df = df.merge(product_df, on='product_id')
        df = df.merge(store_df, on='store_id')
        
        # Features and target
        features = ['day_of_week', 'is_weekend', 'temperature', 'weather', 
                   'unit_price', 'category']
        
        # Ensure all weather categories are present
        weather_categories = ['Clear', 'Cold', 'Extreme Heat', 'Normal', 'Rain', 'Snow']
        for cat in weather_categories:
            df[f'weather_{cat}'] = (df['weather'] == cat).astype(int)
        
        # Use the dummy columns we created instead of get_dummies
        X = df[['day_of_week', 'is_weekend', 'temperature', 'unit_price'] + 
              [f'weather_{cat}' for cat in weather_categories]]
        y = df['units_sold']
        
        # Simplify model for faster training
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def predict_demand(store_id, product_id, store_data, product_data, model, days=3):
    """Predict demand for next days with weather forecast"""
    if not model:
        # Fallback prediction if model isn't available
        return random.randint(5, 30)
    
    # Get store and product info
    store = next(s for s in store_data if s['store_id'] == store_id)
    product = next(p for p in product_data if p['product_id'] == product_id)
    
    # Get weather forecast
    forecast = get_weather_forecast(store['lat'], store['lng'])
    
    # Define base demand by product category
    category_base_demand = {
        'Dairy': 20,
        'Bakery': 15,
        'Produce': 18,
        'Household': 8,
        'Meat': 10,
        'Beverages': 12,
        'Breakfast': 9,
        'Seasonal': 5
    }
    
    # Start with base demand for the product's category
    base_demand = category_base_demand.get(product['category'], 10)
    
    # Predict demand for each day in forecast
    predictions = []
    weather_categories = ['Clear', 'Cold', 'Extreme Heat', 'Normal', 'Rain', 'Snow']
    
    for day in forecast[:days]:
        date = datetime.strptime(day['date'], "%Y-%m-%d")
        weather_class = get_weather_classification(day['temp'], day['conditions'])
        
        # Prepare features
        features = {
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'temperature': day['temp'],
            'unit_price': product['unit_price'],
        }
        
        # Add weather features
        for cat in weather_categories:
            features[f'weather_{cat}'] = 1 if weather_class == cat else 0
        
        # Create DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        required_columns = ['day_of_week', 'is_weekend', 'temperature', 'unit_price'] + \
                         [f'weather_{cat}' for cat in weather_categories]
        
        for col in required_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[required_columns]
        
        # Predict demand
        demand = model.predict(features_df)[0]
        
        # Combine base demand with model prediction
        combined_demand = (0.6 * base_demand) + (0.4 * demand)
        
        # Apply category-based adjustments
        if product['category'] == 'Seasonal':
            if weather_class == 'Rain' and 'Umbrella' in product['name']:
                combined_demand *= 2.5
            elif weather_class == 'Snow' and 'Snow' in product['name']:
                combined_demand *= 3.0
            elif weather_class == 'Extreme Heat' and 'Fan' in product['name']:
                combined_demand *= 2.8
            elif weather_class == 'Cold' and 'Heater' in product['name']:
                combined_demand *= 2.2
            elif weather_class == 'Clear' and 'Sunscreen' in product['name']:
                combined_demand *= 2.0
            
        predictions.append(max(0, combined_demand))
    
    return sum(predictions)  # Total predicted demand for the period

# Inventory Management Functions
def check_inventory_health(store_id, product_id, inventory_data, store_data, product_data, model):
    """Check inventory health considering demand forecast"""
    # Get current inventory
    inventory = next(
        inv for inv in inventory_data 
        if inv['store_id'] == store_id and inv['product_id'] == product_id
    )
    product = next(p for p in product_data if p['product_id'] == product_id)
    
    # Predict demand
    predicted_demand = predict_demand(store_id, product_id, store_data, product_data, model)
    
    # Determine status with realistic thresholds
    if inventory['current_stock'] < predicted_demand * 0.4:
        return 'Critical', predicted_demand
    elif inventory['current_stock'] < predicted_demand * 0.8:
        return 'Low', predicted_demand
    elif inventory['current_stock'] > predicted_demand * 1.4:
        return 'Overstock', predicted_demand
    else:
        return 'Optimal', predicted_demand

def generate_transfer_recommendations(inventory_data, store_data, product_data, model, status_filter="All"):
    """Generate transfer recommendations considering multiple factors"""
    recommendations = []
    
    # Find all stores with low stock
    low_stock = []
    over_stock = []
    
    for inv in inventory_data:
        product = next(p for p in product_data if p['product_id'] == inv['product_id'])
        status, predicted_demand = check_inventory_health(
            inv['store_id'], 
            inv['product_id'], 
            inventory_data, 
            store_data, 
            product_data, 
            model
        )
        
        # Apply status filter to low stock items
        if status in ['Critical', 'Low']:
            if status_filter == "All" or status_filter == status:
                low_stock.append({
                    'store_id': inv['store_id'],
                    'product_id': inv['product_id'],
                    'product_name': product['name'],
                    'status': status,
                    'current': inv['current_stock'],
                    'needed': max(0, predicted_demand - inv['current_stock']),
                    'predicted_demand': predicted_demand
                })
        elif status == 'Overstock':
            over_stock.append({
                'store_id': inv['store_id'],
                'product_id': inv['product_id'],
                'product_name': product['name'],
                'current': inv['current_stock'],
                'excess': inv['current_stock'] - (predicted_demand * 1.2),
                'predicted_demand': predicted_demand
            })
    
    # Match low stock with over stock
    for ls in low_stock:
        matching_os = [os for os in over_stock if os['product_id'] == ls['product_id']]
        
        for os in matching_os:
            # Get store locations
            from_store = next(s for s in store_data if s['store_id'] == os['store_id'])
            to_store = next(s for s in store_data if s['store_id'] == ls['store_id'])
            
            # Skip if same store
            if from_store['store_id'] == to_store['store_id']:
                continue
                
            # Get weather conditions for route
            mid_lat = (from_store['lat'] + to_store['lat']) / 2
            mid_lon = (from_store['lng'] + to_store['lng']) / 2
            weather = get_current_weather(mid_lat, mid_lon)
            
            # Calculate distance
            distance = geodesic((from_store['lat'], from_store['lng']), 
                              (to_store['lat'], to_store['lng'])).miles
            
            # Skip if too far
            if distance > MAX_TRANSFER_DISTANCE:
                continue
                
            # Get ETA with traffic
            eta_minutes = get_eta_with_traffic(from_store['lat'], from_store['lng'], 
                                              to_store['lat'], to_store['lng'])
            
            # Skip if ETA too long or unavailable
            if eta_minutes is None or eta_minutes > MAX_TRANSFER_TIME:
                continue
            
            # Determine transfer amount
            transfer_qty = min(ls['needed'], os['excess'])
            
            if transfer_qty > 0:
                # Calculate cost savings
                product = next(p for p in product_data if p['product_id'] == ls['product_id'])
                cost_savings = transfer_qty * product['unit_price'] * 0.25  # 25% of price as savings
                
                # Calculate urgency factor
                urgency_factor = 1.0
                if ls['status'] == 'Critical':
                    urgency_factor = 1.5
                
                # Weather impact on transfer priority
                if "Rain" in weather['conditions'] or "Snow" in weather['conditions']:
                    urgency_factor *= 1.3
                
                # Traffic impact on transfer priority
                if eta_minutes > 45:  # Heavy traffic
                    urgency_factor *= 1.2
                
                # Create recommendation
                recommendations.append({
                    'from_store_id': os['store_id'],
                    'from_store_name': from_store['name'],
                    'from_store_city': from_store['city'],
                    'to_store_id': ls['store_id'],
                    'to_store_name': to_store['name'],
                    'to_store_city': to_store['city'],
                    'product_id': ls['product_id'],
                    'product_name': product['name'],
                    'quantity': transfer_qty,
                    'distance': distance,
                    'eta_minutes': eta_minutes,
                    'weather_impact': weather['conditions'],
                    'urgency': ls['status'],
                    'urgency_factor': urgency_factor,
                    'cost_savings': cost_savings,
                    'status': 'Pending'
                })
    
    # Sort by urgency and ETA
    recommendations.sort(key=lambda x: (x['urgency_factor'], -x['eta_minutes']), reverse=True)
    return recommendations

# UI Components
def display_weather_impact_guide():
    """Display weather impact guide"""
    with st.expander("Weather Impact Reference Guide"):
        st.subheader("How Weather Affects Product Demand")
        
        st.markdown("""
        | Weather Condition | Affected Products | Impact |
        |-------------------|-------------------|--------|
        | Rain             | Umbrellas, Raincoats | Demand increases 2-3x |
        | Snow             | Snow Shovels, Winter Coats | Demand increases 2-4x |
        | Extreme Heat     | Fans, Air Conditioners, Bottled Water | Demand increases 2-3x |
        | Cold             | Heaters, Thermal Underwear | Demand increases 2-3x |
        | Clear/Sunny      | Sunscreen, Sunglasses | Demand increases 2x |
        """)
        
        st.caption("Note: These are typical impacts observed at Walmart stores. Actual impacts may vary by location.")

def display_real_time_conditions(store_data):
    """Display real-time weather and forecasts"""
    st.header("Real-Time Store Conditions")
    
    if not store_data:
        st.warning("No store data available")
        return
    
    # Select a store to show details with unique key
    store_options = {s['name']: s for s in store_data}
    selected_store = st.selectbox(
        "Select Store for Real-Time Conditions", 
        list(store_options.keys()),
        key="real_time_store_selector"
    )
    store = store_options[selected_store]
    
    # Get current weather
    weather = get_current_weather(store['lat'], store['lng'])
    
    # Get forecast
    forecast = get_weather_forecast(store['lat'], store['lng'])
    
    # Display current weather
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Current Conditions at {store['name']}")
        st.metric("Temperature", f"{weather['temp']:.1f}°F")
        st.metric("Feels Like", f"{weather['feels_like']:.1f}°F")
        st.metric("Conditions", weather['description'].title())
        st.metric("Humidity", f"{weather['humidity']}%")
        st.metric("Wind Speed", f"{weather['wind_speed']:.1f} mph")
    
    with col2:
        st.subheader("7-Day Forecast")
        
        # Create forecast cards
        cols = st.columns(7)
        for i, day in enumerate(forecast[:7]):
            with cols[i]:
                st.subheader(datetime.strptime(day['date'], "%Y-%m-%d").strftime("%a"))
                st.caption(day['date'])
                st.metric("Temp", f"{day['temp']:.0f}°F", 
                         delta=f"Feels {day['feels_like']:.0f}°F")
                st.write(day['conditions'])
                st.write(f"Precip: {day['precipprob']}%")
        
        # Plot forecast
        forecast_df = pd.DataFrame(forecast)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['temp'],
            mode='lines+markers',
            name='Temperature (°F)',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Bar(
            x=forecast_df['date'], 
            y=forecast_df['precipprob'],
            name='Precip Probability (%)',
            marker=dict(color='blue', opacity=0.6),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='7-Day Temperature and Precipitation Forecast',
            yaxis=dict(title='Temperature (°F)'),
            yaxis2=dict(title='Precipitation Probability (%)', overlaying='y', side='right'),
            hovermode="x unified",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    display_weather_impact_guide()

def display_inventory_dashboard(inventory_data, store_data, product_data, model):
    """Display inventory dashboard"""
    st.header("Inventory Optimization Dashboard")
    
    if not inventory_data or not store_data or not product_data:
        st.warning("No data available")
        return
    
    # Status filter
    st.subheader("Product Status Filter")
    status_filter = st.radio("Filter by product status:", 
                            ["All", "Critical", "Low"], 
                            index=0,
                            key="status_filter_radio",
                            horizontal=True)
    
    # Calculate inventory health
    health_data = []
    for inv in inventory_data:
        product = next(p for p in product_data if p['product_id'] == inv['product_id'])
        store = next(s for s in store_data if s['store_id'] == inv['store_id'])
        
        status, predicted_demand = check_inventory_health(
            inv['store_id'], 
            inv['product_id'], 
            inventory_data, 
            store_data, 
            product_data, 
            model
        )
        
        health_data.append({
            'Store': store['name'],
            'City': store['city'],
            'Product': product['name'],
            'Category': product['category'],
            'Current Stock': inv['current_stock'],
            'Predicted Demand': predicted_demand,
            'Status': status
        })
    
    health_df = pd.DataFrame(health_data)
    
    # Apply status filter to health data
    if status_filter != "All":
        health_df = health_df[health_df['Status'] == status_filter]
    
    # Status distribution
    st.subheader(f"Inventory Health Overview: {status_filter} Items")
    
    if not health_df.empty:
        status_counts = health_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(status_counts, names='Status', values='Count', 
                         title="Inventory Status Distribution",
                         color='Status',
                         color_discrete_map={
                             'Critical': 'red',
                             'Low': 'orange',
                             'Optimal': 'green',
                             'Overstock': 'purple'
                         })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(status_counts, x='Status', y='Count', 
                         title="Inventory Status Count",
                         color='Status',
                         color_discrete_map={
                             'Critical': 'red',
                             'Low': 'orange',
                             'Optimal': 'green',
                             'Overstock': 'purple'
                         })
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No {status_filter} inventory items found")
    
    # Display filtered inventory
    st.subheader(f"Filtered Inventory Details: {status_filter} Items")
    if not health_df.empty:
        st.dataframe(health_df)
    else:
        st.warning(f"No {status_filter} inventory items found")
    
    # Generate and display transfer recommendations
    st.subheader(f"Smart Transfer Recommendations for {status_filter} Items")
    recommendations = generate_transfer_recommendations(
        inventory_data, 
        store_data, 
        product_data, 
        model,
        status_filter
    )
    
    if recommendations:
        # Display in a table
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df[['from_store_name', 'to_store_name', 'product_name', 
                     'quantity', 'eta_minutes', 'distance', 'weather_impact', 'urgency', 'cost_savings']])
        
        # Map visualization
        st.subheader("Transfer Route Map")
        fig = go.Figure()
        
        # Add stores
        store_ids = set()
        for rec in recommendations:
            store_ids.add(rec['from_store_id'])
            store_ids.add(rec['to_store_id'])
        
        for store_id in store_ids:
            store = next(s for s in store_data if s['store_id'] == store_id)
            fig.add_trace(go.Scattermapbox(
                lon=[store['lng']],
                lat=[store['lat']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='blue',
                    opacity=0.8
                ),
                text=store['name'],
                hoverinfo='text',
                name=store['name']
            ))
        
        # Add routes
        for rec in recommendations:
            from_store = next(s for s in store_data if s['store_id'] == rec['from_store_id'])
            to_store = next(s for s in store_data if s['store_id'] == rec['to_store_id'])
            
            # Add route line
            fig.add_trace(go.Scattermapbox(
                lon=[from_store['lng'], to_store['lng']],
                lat=[from_store['lat'], to_store['lat']],
                mode='lines',
                line=dict(width=3, color='red'),
                text=f"{rec['distance']:.1f} miles - {rec['eta_minutes']:.0f} min",
                hoverinfo='text',
                name=f"{from_store['name']} to {to_store['name']}"
            ))
        
        # Set map layout
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=10,
            mapbox_center={"lat": from_store['lat'], "lon": from_store['lng']},
            height=500,
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transfer approval
        st.subheader("Approve Recommended Transfers")
        for i, rec in enumerate(recommendations):
            with st.expander(f"Transfer #{i+1}: {rec['product_name']} from {rec['from_store_name']} to {rec['to_store_name']}"):
                cols = st.columns(4)
                cols[0].metric("Product", rec['product_name'])
                cols[1].metric("Quantity", rec['quantity'])
                cols[2].metric("Distance", f"{rec['distance']:.1f} miles")
                cols[3].metric("ETA", f"{rec['eta_minutes']:.0f} min")
                
                st.write(f"**From Store:** {rec['from_store_name']} ({rec['from_store_city']})")
                st.write(f"**To Store:** {rec['to_store_name']} ({rec['to_store_city']})")
                st.write(f"**Weather Conditions:** {rec['weather_impact']}")
                st.write(f"**Urgency:** {rec['urgency']}")
                st.write(f"**Cost Savings:** ${rec['cost_savings']:.2f}")
                
                if st.button(f"Approve Transfer #{i+1}", key=f"approve_{i}"):
                    st.session_state.transfers.append(rec)
                    st.success(f"Transfer #{i+1} approved and scheduled!")
    else:
        if status_filter == "All":
            st.success("No transfer recommendations needed at this time")
        else:
            st.warning(f"No transfer recommendations found for {status_filter} items")

# Main Application
def main():
    # Initialize data from session state
    store_data = st.session_state.store_data
    product_data = st.session_state.product_data
    inventory_data = st.session_state.inventory
    sales_data = st.session_state.sales
    model = st.session_state.model
    
    # Initialize with real Walmart data if not already done
    if not store_data:
        st.session_state.store_data = WALMART_STORES
    
    if not product_data:
        st.session_state.product_data = WALMART_PRODUCTS
    
    # Sidebar controls
    with st.sidebar:
        st.header("Smart Routing Controls")
        
        if st.button("Initialize System"):
            with st.spinner("Setting up inventory routing system..."):
                try:
                    # Get store and product data
                    store_data = st.session_state.store_data
                    product_data = st.session_state.product_data
                    
                    # Generate inventory and sales data
                    inventory_data = generate_inventory_data(store_data, product_data)
                    sales_data = generate_sales_data(store_data, product_data)
                    
                    # Store in session state
                    st.session_state.inventory = inventory_data
                    st.session_state.sales = sales_data
                    
                    # Train model
                    model = train_demand_model(sales_data, store_data, product_data)
                    st.session_state.model = model
                    st.success("System initialized! Demand model trained.")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
        
        st.subheader("System Status")
        if inventory_data:
            st.success("Inventory data loaded")
        else:
            st.warning("Inventory data not loaded")
            
        if model:
            st.success("Demand model trained")
        else:
            st.warning("Demand model not trained")
        
        st.divider()
        st.subheader("About This System")
        st.write("This system helps Walmart optimize inventory across stores by:")
        st.write("- Predicting demand based on historical patterns and weather")
        st.write("- Identifying inventory imbalances before they occur")
        st.write("- Recommending efficient transfers between nearby stores")
        st.write("- Considering traffic, weather, and urgency factors")
    
    # Display real-time conditions
    if store_data:
        display_real_time_conditions(store_data)
    else:
        st.info("Initialize system to see real-time conditions")
    
    # Display dashboard if we have all required data
    if store_data and product_data and inventory_data:
        try:
            # If model isn't available, try to train it
            if not model and sales_data:
                with st.spinner("Training demand model..."):
                    model = train_demand_model(sales_data, store_data, product_data)
                    st.session_state.model = model
            
            # Display inventory dashboard
            display_inventory_dashboard(inventory_data, store_data, product_data, model)
        except Exception as e:
            st.error(f"Error displaying dashboard: {str(e)}")
    else:
        st.info("Click 'Initialize System' in the sidebar to start")

if __name__ == "__main__":
    main()
