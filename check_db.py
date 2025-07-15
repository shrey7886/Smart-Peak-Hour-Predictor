import sqlite3

try:
    conn = sqlite3.connect('/app/shop_profiles.db')
    cursor = conn.cursor()
    
    # Check what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables in database:", tables)
    
    # Check shop_profiles table
    cursor.execute("SELECT * FROM shop_profiles")
    shops = cursor.fetchall()
    print("Shops in shop_profiles table:", shops)
    
    # Check table structure
    cursor.execute("PRAGMA table_info(shop_profiles)")
    columns = cursor.fetchall()
    print("Shop_profiles table structure:", columns)
        
    conn.close()
except Exception as e:
    print("Error:", e) 