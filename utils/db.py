import sqlite3
import json
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection."""
    try:
        conn = sqlite3.connect('shop_profiles.db')
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_db():
    """Initialize the database with required tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create shop profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shop_profiles (
                shop_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                last_upload TIMESTAMP,
                model_path TEXT,
                forecast_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create shop forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shop_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shop_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                forecast_csv TEXT NOT NULL,
                peak_hours TEXT,
                recommendations TEXT,
                FOREIGN KEY (shop_id) REFERENCES shop_profiles (shop_id)
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        conn.close()

def save_shop_profile(shop_id: str, name: str, location: str) -> None:
    """Save or update shop profile in the database."""
    try:
        # Create shop profile with all required fields
        shop = {
            'shop_id': shop_id,
            'name': name,  # Ensure name is included
            'location': location,
            'created_at': datetime.now().isoformat(),
            'last_upload': None,
            'settings': {
                'timezone': 'Asia/Kolkata',
                'business_hours': {
                    'start': '09:00',
                    'end': '21:00'
                }
            }
        }
        
        # Create directories for the shop
        create_shop_directories(shop_id)
        
        # Save to database
        db_path = os.path.join('data', 'shops', shop_id, 'profile.json')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, 'w') as f:
            json.dump(shop, f, indent=4)
            
        logger.info(f"Shop profile saved/updated: {shop_id}")
        
    except Exception as e:
        logger.error(f"Error saving shop profile: {str(e)}")
        raise

def get_shop_profile(shop_id):
    """Retrieve a shop profile by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shop_profiles WHERE shop_id = ?", (shop_id,))
        profile = cursor.fetchone()
        return dict(profile) if profile else None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving shop profile: {e}")
        raise
    finally:
        conn.close()

def save_forecast(shop_id, forecast_csv_path, peak_hours, recommendations):
    """Save a new forecast for a shop."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert peak hours and recommendations to JSON strings
        peak_hours_json = json.dumps(peak_hours)
        recommendations_json = json.dumps(recommendations)
        
        cursor.execute('''
            INSERT INTO shop_forecasts 
            (shop_id, forecast_csv, peak_hours, recommendations)
            VALUES (?, ?, ?, ?)
        ''', (shop_id, forecast_csv_path, peak_hours_json, recommendations_json))
        
        # Update last_upload in shop_profiles
        cursor.execute('''
            UPDATE shop_profiles 
            SET last_upload = ?, updated_at = ?
            WHERE shop_id = ?
        ''', (datetime.now().isoformat(), datetime.now().isoformat(), shop_id))
        
        conn.commit()
        logger.info(f"Forecast saved for shop: {shop_id}")
    except sqlite3.Error as e:
        logger.error(f"Error saving forecast: {e}")
        raise
    finally:
        conn.close()

def get_forecast_history(shop_id, limit=5):
    """Retrieve forecast history for a shop."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, forecast_csv, peak_hours, recommendations
            FROM shop_forecasts
            WHERE shop_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (shop_id, limit))
        
        rows = cursor.fetchall()
        forecasts = []
        for row in rows:
            forecast = dict(row)
            # Parse JSON strings back to Python objects
            forecast['peak_hours'] = json.loads(forecast['peak_hours'])
            forecast['recommendations'] = json.loads(forecast['recommendations'])
            forecasts.append(forecast)
        
        return forecasts
    except sqlite3.Error as e:
        logger.error(f"Error retrieving forecast history: {e}")
        raise
    finally:
        conn.close()

def get_all_shops():
    """Retrieve all shop profiles."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shop_profiles ORDER BY updated_at DESC")
        shops = [dict(row) for row in cursor.fetchall()]
        return shops
    except sqlite3.Error as e:
        logger.error(f"Error retrieving all shops: {e}")
        raise
    finally:
        conn.close()

def create_shop_directories(shop_id):
    """Create necessary directories for a shop's data."""
    directories = [
        f"models/{shop_id}",
        f"data/{shop_id}",
        f"forecasts/{shop_id}"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directories for shop: {shop_id}")

class DatabaseManager:
    """A class to manage database operations with a higher-level interface."""
    
    def __init__(self):
        """Initialize the database manager."""
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create shop profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shop_profiles (
                    shop_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location TEXT,
                    last_upload TIMESTAMP,
                    model_path TEXT,
                    forecast_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create shop forecasts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shop_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shop_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    forecast_csv TEXT NOT NULL,
                    peak_hours TEXT,
                    recommendations TEXT,
                    FOREIGN KEY (shop_id) REFERENCES shop_profiles (shop_id)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def save_shop(self, shop_id: str, name: str, location: str, **kwargs) -> None:
        """Save or update a shop profile with additional metadata."""
        try:
            # Create shop profile with all required fields
            shop = {
                'shop_id': shop_id,
                'name': name,
                'location': location,
                'created_at': datetime.now().isoformat(),
                'last_upload': None,
                'settings': {
                    'timezone': kwargs.get('timezone', 'Asia/Kolkata'),
                    'business_hours': kwargs.get('business_hours', {
                        'start': '09:00',
                        'end': '21:00'
                    })
                },
                **kwargs
            }
            
            # Create directories for the shop
            create_shop_directories(shop_id)
            
            # Save to database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO shop_profiles 
                (shop_id, name, location, created_at, last_upload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                shop_id,
                name,
                location,
                shop['created_at'],
                shop['last_upload'],
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            logger.info(f"Shop profile saved/updated: {shop_id}")
            
            # Save additional metadata to JSON file
            db_path = os.path.join('data', 'shops', shop_id, 'profile.json')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with open(db_path, 'w') as f:
                json.dump(shop, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving shop profile: {str(e)}")
            raise
    
    def get_shop(self, shop_id: str) -> dict:
        """Get a shop profile by ID."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM shop_profiles WHERE shop_id = ?", (shop_id,))
            profile = cursor.fetchone()
            
            if not profile:
                return None
                
            # Convert to dict and load additional metadata
            shop = dict(profile)
            db_path = os.path.join('data', 'shops', shop_id, 'profile.json')
            
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    metadata = json.load(f)
                    shop.update(metadata)
            
            return shop
            
        except Exception as e:
            logger.error(f"Error retrieving shop profile: {str(e)}")
            raise
    
    def save_forecast(self, shop_id: str, forecast_data: dict) -> None:
        """Save a forecast with all associated data."""
        try:
            cursor = self.conn.cursor()
            
            # Convert data to JSON strings
            forecast_csv = json.dumps(forecast_data.get('forecast_csv', {}))
            peak_hours = json.dumps(forecast_data.get('peak_hours', []))
            recommendations = json.dumps(forecast_data.get('recommendations', []))
            
            # Save forecast
            cursor.execute('''
                INSERT INTO shop_forecasts 
                (shop_id, forecast_csv, peak_hours, recommendations)
                VALUES (?, ?, ?, ?)
            ''', (shop_id, forecast_csv, peak_hours, recommendations))
            
            # Update shop's last upload time
            cursor.execute('''
                UPDATE shop_profiles 
                SET last_upload = ?, updated_at = ?
                WHERE shop_id = ?
            ''', (datetime.now().isoformat(), datetime.now().isoformat(), shop_id))
            
            self.conn.commit()
            logger.info(f"Forecast saved for shop: {shop_id}")
            
        except Exception as e:
            logger.error(f"Error saving forecast: {str(e)}")
            raise
    
    def get_forecasts(self, shop_id: str, limit: int = 5) -> list:
        """Get forecast history for a shop."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT timestamp, forecast_csv, peak_hours, recommendations
                FROM shop_forecasts
                WHERE shop_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (shop_id, limit))
            
            forecasts = []
            for row in cursor.fetchall():
                forecast = dict(row)
                # Parse JSON strings back to Python objects
                forecast['forecast_csv'] = json.loads(forecast['forecast_csv'])
                forecast['peak_hours'] = json.loads(forecast['peak_hours'])
                forecast['recommendations'] = json.loads(forecast['recommendations'])
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {str(e)}")
            raise
    
    def get_all_shops(self) -> list:
        """Get all shop profiles with their metadata."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM shop_profiles ORDER BY updated_at DESC")
            
            shops = []
            for row in cursor.fetchall():
                shop = dict(row)
                # Load additional metadata
                db_path = os.path.join('data', 'shops', shop['shop_id'], 'profile.json')
                if os.path.exists(db_path):
                    with open(db_path, 'r') as f:
                        metadata = json.load(f)
                        shop.update(metadata)
                shops.append(shop)
            
            return shops
            
        except Exception as e:
            logger.error(f"Error retrieving all shops: {str(e)}")
            raise 