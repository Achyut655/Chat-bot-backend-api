"""
Financial Chatbot Microservice with Groq
----------------------------------------
A lightweight Flask API for a financial chatbot that uses Groq's LLM capabilities.
Adapted for AWS Lambda with Python 3.13
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import datetime
import json
import pymysql
import decimal

from groq import Groq
import uuid
import awsgi  # AWS-specific adapter for WSGI applications

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'user'),
    'password': os.environ.get('DB_PASSWORD', 'password'),
    'database': os.environ.get('DB_NAME', 'finance_app')
}

# Groq API setup
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
groq_client = Groq(api_key=GROQ_API_KEY)

# Create necessary tables for chat functionality
def setup_chat_schema():
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()
    
    # Create chats table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS `chats` (
      `chat_id` varchar(36) NOT NULL,
      `user_id` bigint NOT NULL,
      `title` varchar(255) DEFAULT NULL,
      `created_at` datetime(6) NOT NULL,
      `updated_at` datetime(6) NOT NULL,
      `is_active` bit(1) DEFAULT b'1',
      PRIMARY KEY (`chat_id`),
      KEY `FK_chats_users` (`user_id`),
      CONSTRAINT `FK_chats_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
    )
    """)
    
    # Create chat_messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS `chat_messages` (
      `message_id` varchar(36) NOT NULL,
      `chat_id` varchar(36) NOT NULL,
      `content` text NOT NULL,
      `is_user_message` bit(1) NOT NULL,
      `created_at` datetime(6) NOT NULL,
      PRIMARY KEY (`message_id`),
      KEY `FK_chat_messages_chats` (`chat_id`),
      CONSTRAINT `FK_chat_messages_chats` FOREIGN KEY (`chat_id`) REFERENCES `chats` (`chat_id`)
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

# Lambda doesn't use app.before_first_request, so we need a different approach
def initialize_db_if_needed():
    # Only run schema setup in Lambda cold starts or when needed
    try:
        setup_chat_schema()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")

# Helper functions for database operations
def execute_query(query, params=None, fetchone=False):
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    
    result = None
    if fetchone:
        result = cursor.fetchone()
    else:
        result = cursor.fetchall()
    
    conn.commit()
    cursor.close()
    conn.close()
    return result

def fetch_user_data(user_id):
    """Fetch relevant financial data for the user to provide context to the LLM"""
    # Get user's transactions
    transactions = execute_query(
        "SELECT amount, merchant_name, created_at, carbon_emission FROM transactions WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
        (user_id,)
    )
    
    # Convert datetime to string
    for transaction in transactions:
        transaction['created_at'] = transaction['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Get user's categories and spending
    categories = execute_query(
        """
        SELECT c.name, SUM(t.amount) as total
        FROM transactions t
        JOIN categories c ON t.category_id = c.id
        WHERE t.user_id = %s
        GROUP BY c.name
        """,
        (user_id,)
    )
    
    # Format data for context
    context = {
        "recent_transactions": transactions,
        "category_spending": categories
    }
    
    return context

# Custom JSON encoder to handle Decimal objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = CustomJSONEncoder

# Routes for chat functionality
@app.route('/api/chats/start', methods=['POST'])
def start_chat():
    # Initialize DB if needed (for Lambda cold starts)
    initialize_db_if_needed()
    
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Check if user exists
    user = execute_query("SELECT id FROM users WHERE id = %s", (user_id,), fetchone=True)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Generate unique chat ID
    chat_id = str(uuid.uuid4())
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Create new chat
    execute_query(
        "INSERT INTO chats (chat_id, user_id, title, created_at, updated_at, is_active) VALUES (%s, %s, %s, %s, %s, 1)",
        (chat_id, user_id, "New Financial Chat", now, now)
    )
    
    # Add initial system message
    system_message = "Hello! I'm your financial assistant. How can I help you with your finances today?"
    message_id = str(uuid.uuid4())
    
    execute_query(
        "INSERT INTO chat_messages (message_id, chat_id, content, is_user_message, created_at) VALUES (%s, %s, %s, %s, %s)",
        (message_id, chat_id, system_message, 0, now)
    )
    
    return jsonify({
        "chat_id": chat_id,
        "message": system_message
    })

@app.route('/api/chats/<chat_id>/message', methods=['POST'])
def send_message(chat_id):
    # Initialize DB if needed (for Lambda cold starts)
    initialize_db_if_needed()
    
    data = request.json
    user_message = data.get('message')
    user_id = data.get('user_id')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Check if chat exists and belongs to user
    chat = execute_query(
        "SELECT * FROM chats WHERE chat_id = %s AND user_id = %s AND is_active = 1",
        (chat_id, user_id),
        fetchone=True
    )
    
    if not chat:
        return jsonify({"error": "Chat not found or inactive"}), 404
    
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    print(f"User message: {user_message}")
    print(f"Chat ID: {chat_id}")
    print(f"User ID: {user_id}")

    # Save user message
    user_message_id = str(uuid.uuid4())
    execute_query(
        "INSERT INTO chat_messages (message_id, chat_id, content, is_user_message, created_at) VALUES (%s, %s, %s, %s, %s)",
        (user_message_id, chat_id, user_message, 1, now)
    )
    
    # Get chat history for context
    messages = execute_query(
        "SELECT content, is_user_message FROM chat_messages WHERE chat_id = %s ORDER BY created_at ASC",
        (chat_id,)
    )
    
    # Get user financial data for context
    user_data = fetch_user_data(user_id)

    # Format messages for Groq
    formatted_messages = [
        {"role": "system", "content": f"""You are a helpful financial assistant for a personal finance tracking application.
        You have access to the user's financial data and can answer questions about their spending, budget, and carbon footprint.
        Current user data: {json.dumps(user_data, cls=CustomJSONEncoder)}
        
        Be helpful, concise, and focus on providing actionable financial insights."""}
    ]
    
    for msg in messages:
        role = "user" if msg['is_user_message'] else "assistant"
        formatted_messages.append({"role": role, "content": msg['content']})
    
    # Get response from Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # or another Groq model
            messages=formatted_messages
        )
        assistant_response = response.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        assistant_response = "I'm sorry, I'm having trouble processing your request right now. Please try again later."
    
    # Save assistant response
    assistant_message_id = str(uuid.uuid4())
    execute_query(
        "INSERT INTO chat_messages (message_id, chat_id, content, is_user_message, created_at) VALUES (%s, %s, %s, %s, %s)",
        (assistant_message_id, chat_id, assistant_response, 0, now)
    )
    
    # Update chat timestamp
    execute_query(
        "UPDATE chats SET updated_at = %s WHERE chat_id = %s",
        (now, chat_id)
    )
    
    return jsonify({
        "message_id": assistant_message_id,
        "content": assistant_response
    })


@app.route('/api/chats/<chat_id>/end', methods=['POST'])
def end_chat(chat_id):
    # Initialize DB if needed (for Lambda cold starts)
    initialize_db_if_needed()
    
    data = request.json
    user_id = data.get('user_id')
    
    # Verify chat exists and belongs to user
    chat = execute_query(
        "SELECT * FROM chats WHERE chat_id = %s AND user_id = %s AND is_active = 1",
        (chat_id, user_id),
        fetchone=True
    )
    
    if not chat:
        return jsonify({"error": "Chat not found or already inactive"}), 404
    
    # Mark chat as inactive
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    execute_query(
        "UPDATE chats SET is_active = 0, updated_at = %s WHERE chat_id = %s",
        (now, chat_id)
    )
    
    return jsonify({"success": True, "message": "Chat ended successfully"})

@app.route('/api/chats', methods=['GET'])
def get_user_chats():
    # Initialize DB if needed (for Lambda cold starts)
    initialize_db_if_needed()
    
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Get all chats for user
    chats = execute_query(
        """
        SELECT c.chat_id, c.title, c.created_at, c.updated_at, c.is_active,
               (SELECT content FROM chat_messages 
                WHERE chat_id = c.chat_id ORDER BY created_at DESC LIMIT 1) as last_message
        FROM chats c
        WHERE c.user_id = %s
        ORDER BY c.updated_at DESC
        """,
        (user_id,)
    )
    
    return jsonify({"chats": chats})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    # Initialize DB if needed (for Lambda cold starts)
    initialize_db_if_needed()
    
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Verify chat belongs to user
    chat = execute_query(
        "SELECT * FROM chats WHERE chat_id = %s AND user_id = %s",
        (chat_id, user_id),
        fetchone=True
    )
    
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    
    # Get all messages for chat
    messages = execute_query(
        "SELECT message_id, content, is_user_message, created_at FROM chat_messages WHERE chat_id = %s ORDER BY created_at ASC",
        (chat_id,)
    )
    
    return jsonify({
        "chat_id": chat_id,
        "title": chat['title'],
        "is_active": chat['is_active'],
        "messages": messages
    })

# AWS Lambda handler function
def lambda_handler(event, context):
    return awsgi.response(app, event, context)


if __name__ == '__main__':
    # This is used when running locally for testing
    initialize_db_if_needed()
    app.run(debug=True, host='0.0.0.0', port=5000)