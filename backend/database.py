"""
Database module for Smart Checkout.
Handles SQLite connection and product price operations.
"""

from __future__ import annotations

import sqlite3
import os
from typing import Optional, List

DB_PATH = os.path.join(os.path.dirname(__file__), "products.db")


def get_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create the products table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            price REAL NOT NULL,
            currency TEXT DEFAULT 'AED'
        )
    """)
    conn.commit()

    # Seed default products if table is empty
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        default_products = [
            ("bounty",   "Bounty",           3.50, "AED"),
            ("galaxy",   "Galaxy",            4.00, "AED"),
            ("kitkat",   "KitKat",            3.00, "AED"),
            ("m&m",      "M&M's",             5.00, "AED"),
            ("mars",     "Mars",              3.50, "AED"),
            ("pb_m&m",   "Peanut Butter M&M", 6.00, "AED"),
            ("smarties", "Smarties",          4.50, "AED"),
            ("snickers", "Snickers",          3.75, "AED"),
            ("twix",     "Twix",              3.75, "AED"),
        ]
        cursor.executemany(
            "INSERT INTO products (class_name, display_name, price, currency) VALUES (?, ?, ?, ?)",
            default_products,
        )
        conn.commit()
        print(f"Database seeded with {len(default_products)} products.")

    conn.close()


def get_product_by_class(class_name: str) -> Optional[dict]:
    """Look up a product by its YOLO class name."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT class_name, display_name, price, currency FROM products WHERE class_name = ?",
        (class_name,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_all_products() -> List[dict]:
    """Return all products in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, class_name, display_name, price, currency FROM products ORDER BY display_name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_product_price(class_name: str, price: float) -> bool:
    """Update the price of a product. Returns True if the product was found and updated."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE products SET price = ? WHERE class_name = ?",
        (price, class_name),
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated
