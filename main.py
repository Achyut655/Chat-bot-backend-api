"""
Financial Chatbot Microservice with Groq
----------------------------------------
A lightweight Flask API for a financial chatbot that uses Groq's LLM capabilities.
Enhanced with monthly category-wise data analysis.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import datetime
import json
import pymysql

from groq import Groq
import uuid
from decimal import Decimal
import awsgi

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure CORS with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "https://your-production-domain.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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

# Run schema setup when app starts
@app.before_first_request
def initialize_db():
    setup_chat_schema()

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
    
    # Get monthly category summaries for trends and patterns
    # Fetch data from the last 6 months for analysis
    current_date = datetime.datetime.now()
    current_month_year = current_date.strftime('%Y-%m')
    
    # Get current month for detailed analysis
    current_month_data = execute_query(
        """
        SELECT c.name as category_name, mcs.total_amount, mcs.total_emission
        FROM monthly_category_summaries mcs
        JOIN categories c ON mcs.category_id = c.id
        WHERE mcs.user_id = %s AND mcs.month_year = %s
        ORDER BY mcs.total_amount DESC
        """,
        (user_id, current_month_year)
    )
    
    # Get previous month for comparison
    prev_month_date = current_date.replace(day=1) - datetime.timedelta(days=1)
    prev_month_year = prev_month_date.strftime('%Y-%m')
    
    # Get previous month data
    prev_month_data = execute_query(
        """
        SELECT c.name as category_name, mcs.total_amount
        FROM monthly_category_summaries mcs
        JOIN categories c ON mcs.category_id = c.id
        WHERE mcs.user_id = %s AND mcs.month_year = %s
        """,
        (user_id, prev_month_year)
    )
    
    # Get historical monthly data (6 months)
    monthly_summaries = execute_query(
        """
        SELECT mcs.month_year, c.name as category_name, mcs.total_amount, mcs.total_emission
        FROM monthly_category_summaries mcs
        JOIN categories c ON mcs.category_id = c.id
        WHERE mcs.user_id = %s
        ORDER BY mcs.month_year DESC
        LIMIT 30
        """,
        (user_id,)
    )
    
    # Organize monthly data by month for easier analysis
    monthly_data = {}
    for summary in monthly_summaries:
        month = summary['month_year']
        if month not in monthly_data:
            monthly_data[month] = []
        monthly_data[month].append({
            'category': summary['category_name'],
            'amount': summary['total_amount'],
            'emission': summary['total_emission']
        })
    
    # Calculate month-over-month changes for spending categories
    current_month_insights = []
    prev_month_totals = {item['category_name']: float(item['total_amount']) for item in prev_month_data}
    
    # Calculate current month spending and changes
    total_current_spending = 0
    total_current_emission = 0
    
    for category in current_month_data:
        category_name = category['category_name']
        amount = float(category['total_amount'])
        emission = float(category['total_emission'])
        
        total_current_spending += amount
        total_current_emission += emission
        
        change = 0
        percent_change = 0
        
        if category_name in prev_month_totals:
            prev_amount = prev_month_totals[category_name]
            change = amount - prev_amount
            percent_change = (change / prev_amount) * 100 if prev_amount != 0 else 0
        
        current_month_insights.append({
            'category': category_name,
            'amount': amount,
            'emission': emission,
            'change': change,
            'percent_change': round(percent_change, 2)
        })
    
    # Sort categories by biggest changes (both positive and negative)
    current_month_insights.sort(key=lambda x: abs(x['change']), reverse=True)
    
    # Calculate longer-term trends (past 6 months)
    months = sorted(monthly_data.keys())
    trend_analysis = []
    
    if len(months) > 1:
        for i in range(1, len(months)):
            current_month = months[i]
            prev_month = months[i-1]
            
            # Create lookup dictionaries for current and previous month
            current_data = {item['category']: item['amount'] for item in monthly_data[current_month]}
            prev_data = {item['category']: item['amount'] for item in monthly_data[prev_month]}
            
            # Calculate changes for categories present in both months
            for category in set(current_data.keys()) & set(prev_data.keys()):
                change = float(current_data[category]) - float(prev_data[category])
                percent_change = (change / float(prev_data[category])) * 100 if float(prev_data[category]) != 0 else 0
                
                trend_analysis.append({
                    'from_month': prev_month,
                    'to_month': current_month,
                    'category': category,
                    'change': change,
                    'percent_change': round(percent_change, 2)
                })
    
    # Format data for context
    context = {
        "recent_transactions": transactions,
        "category_spending": categories,
        "monthly_data": monthly_data,
        "current_month": {
            "month_year": current_month_year,
            "total_spending": total_current_spending,
            "total_emission": total_current_emission,
            "category_insights": current_month_insights
        },
        "previous_month": prev_month_year,
        "spending_trends": trend_analysis
    }
    
    return context

# Routes for chat functionality
@app.route('/api/chats/start', methods=['POST'])
def start_chat():
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

    # Convert Decimal objects to float for JSON serialization
    def convert_decimals(obj):
        if isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_decimals(value) for key, value in obj.items()}
        elif isinstance(obj, Decimal):
            return float(obj)
        return obj

    # Convert any Decimals in user_data to floats
    user_data = convert_decimals(user_data)
    print(f"Converted user data: {json.dumps(user_data, indent=2)}")

    # Format messages for Groq
    formatted_messages = [
        {"role": "system", "content": f"""You are a helpful financial assistant for a personal finance tracking application.
        You have access to the user's financial data and can answer questions about their spending, budget, and carbon footprint.
        
        Current user data includes:
        1. Recent transactions (last 20 transactions with amount, merchant, date, and carbon emission)
        2. Overall category-wise spending totals
        3. Monthly spending breakdowns by category (organized by month)
        4. Current month detailed analysis with total spending, emissions, and comparison to previous month
        5. Month-over-month spending trend analysis for each category
        
        User financial data: {json.dumps(user_data)}
        
        Be helpful, concise, and focus on providing actionable financial insights. When the user asks about:
        - Monthly spending: Use the current_month data and monthly_data to provide detailed insights
        - Spending trends: Highlight significant increases/decreases from the current_month.category_insights
        - Carbon emissions: Include data about the user's carbon footprint when relevant
        - Category analysis: Compare spending across different categories and months
        - When suggesting improvements, base them on actual spending patterns visible in the data
        
        If the user asks about a specific month, check if that month exists in the monthly_data keys.
        If the user asks for spending in a specific category, reference both the overall category_spending and the monthly breakdown.
        
        Always provide data-backed insights tailored to this specific user's financial behavior.
        Format currency values appropriately and round percentages to 1-2 decimal places."""}
    ]
    
    for msg in messages:
        role = "user" if msg['is_user_message'] else "assistant"
        formatted_messages.append({"role": role, "content": msg['content']})
    
    # Get response from Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # or another Groq model
            messages=formatted_messages
            # temperature=0.7,
            # max_tokens=800
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



def lambda_handler(event, context):
    return awsgi.response(app, event, context)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
