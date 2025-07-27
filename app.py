from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
from functools import wraps
import os
import io
import tempfile
from datetime import datetime, timedelta
import uuid
import re
from supabase import create_client, Client
import google.generativeai as genai
from google.cloud import speech
import speech_recognition as sr
from pydub import AudioSegment
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS
CORS(app)

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Gemini AI configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database initialization (run this once to create tables)
def init_db():
    """Initialize database tables in Supabase"""
    # Users table
    users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        business_name VARCHAR(200),
        phone VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Products/Inventory table
    products_table = """
    CREATE TABLE IF NOT EXISTS products (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(200) NOT NULL,
        description TEXT,
        sku VARCHAR(100),
        category VARCHAR(100),
        unit_price DECIMAL(10,2) DEFAULT 0,
        quantity INTEGER DEFAULT 0,
        min_stock_level INTEGER DEFAULT 0,
        supplier VARCHAR(200),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Transactions table
    transactions_table = """
    CREATE TABLE IF NOT EXISTS transactions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        transaction_type VARCHAR(20) CHECK (transaction_type IN ('sale', 'expense', 'purchase')),
        amount DECIMAL(10,2) NOT NULL,
        description TEXT,
        category VARCHAR(100),
        date TIMESTAMP DEFAULT NOW(),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Sales table
    sales_table = """
    CREATE TABLE IF NOT EXISTS sales (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
        product_id UUID REFERENCES products(id) ON DELETE SET NULL,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10,2) NOT NULL,
        total_amount DECIMAL(10,2) NOT NULL,
        customer_name VARCHAR(200),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Voice commands log table
    voice_commands_table = """
    CREATE TABLE IF NOT EXISTS voice_commands (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        original_text TEXT,
        processed_command JSON,
        action_taken VARCHAR(100),
        confidence_score DECIMAL(3,2),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

# JWT token decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated

# Voice processing utilities
class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def convert_audio_format(self, audio_file_path):
        """Convert audio to WAV format for better compatibility"""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            wav_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], '.wav')
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            raise Exception(f"Audio conversion failed: {str(e)}")
    
    def speech_to_text(self, audio_file_path):
        """Convert speech to text using speech_recognition library"""
        try:
            wav_path = self.convert_audio_format(audio_file_path)
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            raise Exception(f"Speech to text conversion failed: {str(e)}")
    
    def process_command_with_gemini(self, text):
        """Process natural language command using Gemini AI"""
        prompt = f"""
        You are an AI assistant for inventory and financial management. Parse the following voice command and extract structured data.
        
        Voice command: "{text}"
        
        Please identify the intent and extract relevant information. Respond in JSON format with the following structure:
        {{
            "intent": "add_product|record_sale|record_expense|check_stock|update_stock",
            "confidence": 0.0-1.0,
            "entities": {{
                "product_name": "string",
                "quantity": number,
                "price": number,
                "category": "string",
                "customer_name": "string",
                "description": "string",
                "amount": number
            }},
            "action": "description of what should be done"
        }}
        
        Examples:
        - "Add 50 units of iPhone 15 to inventory at $800 each" -> intent: "add_product"
        - "Record sale of 2 laptops for $1500 to John Smith" -> intent: "record_sale"
        - "I spent $200 on office supplies" -> intent: "record_expense"
        - "Check stock for MacBook Pro" -> intent: "check_stock"
        """
        
        try:
            response = model.generate_content(prompt)
            # Clean up the response to extract JSON
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            parsed_command = json.loads(response_text)
            return parsed_command
        except Exception as e:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "action": f"Failed to process command: {str(e)}"
            }

voice_processor = VoiceProcessor()

# User Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'first_name', 'last_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Check if user already exists
        existing_user = supabase.table('users').select('id').eq('email', data['email']).execute()
        if existing_user.data:
            return jsonify({'error': 'User already exists'}), 409
        
        # Hash password
        password_hash = generate_password_hash(data['password'])
        
        # Create user
        user_data = {
            'email': data['email'],
            'password_hash': password_hash,
            'first_name': data['first_name'],
            'last_name': data['last_name'],
            'business_name': data.get('business_name'),
            'phone': data.get('phone')
        }
        
        result = supabase.table('users').insert(user_data).execute()
        
        if result.data:
            user = result.data[0]
            token = jwt.encode({
                'user_id': user['id'],
                'exp': datetime.utcnow() + timedelta(days=7)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({
                'message': 'User registered successfully',
                'token': token,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name'],
                    'business_name': user['business_name']
                }
            }), 201
        
        return jsonify({'error': 'Registration failed'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Get user from database
        result = supabase.table('users').select('*').eq('email', email).execute()
        
        if not result.data:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = result.data[0]
        
        # Check password
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = jwt.encode({
            'user_id': user['id'],
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'business_name': user['business_name']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Voice Input Routes
@app.route('/api/voice/upload', methods=['POST'])
@token_required
def upload_voice(current_user_id):
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{audio_file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(file_path)
        
        try:
            # Convert speech to text
            text = voice_processor.speech_to_text(file_path)
            
            # Process with Gemini AI
            processed_command = voice_processor.process_command_with_gemini(text)
            
            # Log voice command
            voice_log = {
                'user_id': current_user_id,
                'original_text': text,
                'processed_command': json.dumps(processed_command),
                'confidence_score': processed_command.get('confidence', 0.0)
            }
            supabase.table('voice_commands').insert(voice_log).execute()
            
            # Execute the command based on intent
            action_result = execute_voice_command(current_user_id, processed_command)
            
            return jsonify({
                'original_text': text,
                'processed_command': processed_command,
                'action_result': action_result
            }), 200
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def execute_voice_command(user_id, command):
    """Execute the parsed voice command"""
    intent = command.get('intent')
    entities = command.get('entities', {})
    
    try:
        if intent == 'add_product':
            return add_product_from_voice(user_id, entities)
        elif intent == 'record_sale':
            return record_sale_from_voice(user_id, entities)
        elif intent == 'record_expense':
            return record_expense_from_voice(user_id, entities)
        elif intent == 'check_stock':
            return check_stock_from_voice(user_id, entities)
        elif intent == 'update_stock':
            return update_stock_from_voice(user_id, entities)
        else:
            return {'success': False, 'message': 'Unknown command intent'}
    except Exception as e:
        return {'success': False, 'message': f'Command execution failed: {str(e)}'}

def add_product_from_voice(user_id, entities):
    """Add product from voice command"""
    product_data = {
        'user_id': user_id,
        'name': entities.get('product_name', 'Unknown Product'),
        'description': entities.get('description', ''),
        'unit_price': entities.get('price', 0),
        'quantity': entities.get('quantity', 0),
        'category': entities.get('category', 'General')
    }
    
    result = supabase.table('products').insert(product_data).execute()
    if result.data:
        return {'success': True, 'message': f"Added {product_data['name']} to inventory", 'product': result.data[0]}
    return {'success': False, 'message': 'Failed to add product'}

def record_sale_from_voice(user_id, entities):
    """Record sale from voice command"""
    # Find product by name
    product_name = entities.get('product_name')
    if not product_name:
        return {'success': False, 'message': 'Product name not specified'}
    
    products = supabase.table('products').select('*').eq('user_id', user_id).ilike('name', f'%{product_name}%').execute()
    if not products.data:
        return {'success': False, 'message': f'Product "{product_name}" not found'}
    
    product = products.data[0]
    quantity = entities.get('quantity', 1)
    unit_price = entities.get('price', product['unit_price'])
    total_amount = quantity * unit_price
    
    # Check stock
    if product['quantity'] < quantity:
        return {'success': False, 'message': f'Insufficient stock. Available: {product["quantity"]}'}
    
    # Record transaction
    transaction_data = {
        'user_id': user_id,
        'transaction_type': 'sale',
        'amount': total_amount,
        'description': f'Sale of {quantity} {product["name"]}',
        'category': 'Sales'
    }
    transaction_result = supabase.table('transactions').insert(transaction_data).execute()
    
    if transaction_result.data:
        transaction_id = transaction_result.data[0]['id']
        
        # Record sale
        sale_data = {
            'user_id': user_id,
            'transaction_id': transaction_id,
            'product_id': product['id'],
            'quantity': quantity,
            'unit_price': unit_price,
            'total_amount': total_amount,
            'customer_name': entities.get('customer_name', 'Walk-in Customer')
        }
        sale_result = supabase.table('sales').insert(sale_data).execute()
        
        # Update stock
        new_quantity = product['quantity'] - quantity
        supabase.table('products').update({'quantity': new_quantity}).eq('id', product['id']).execute()
        
        return {
            'success': True,
            'message': f'Recorded sale of {quantity} {product["name"]} for ${total_amount}',
            'sale': sale_result.data[0] if sale_result.data else None
        }
    
    return {'success': False, 'message': 'Failed to record sale'}

def record_expense_from_voice(user_id, entities):
    """Record expense from voice command"""
    amount = entities.get('amount')
    if not amount:
        return {'success': False, 'message': 'Amount not specified'}
    
    transaction_data = {
        'user_id': user_id,
        'transaction_type': 'expense',
        'amount': amount,
        'description': entities.get('description', 'Voice recorded expense'),
        'category': entities.get('category', 'General Expense')
    }
    
    result = supabase.table('transactions').insert(transaction_data).execute()
    if result.data:
        return {'success': True, 'message': f'Recorded expense of ${amount}', 'transaction': result.data[0]}
    return {'success': False, 'message': 'Failed to record expense'}

def check_stock_from_voice(user_id, entities):
    """Check stock from voice command"""
    product_name = entities.get('product_name')
    if not product_name:
        return {'success': False, 'message': 'Product name not specified'}
    
    products = supabase.table('products').select('*').eq('user_id', user_id).ilike('name', f'%{product_name}%').execute()
    if not products.data:
        return {'success': False, 'message': f'Product "{product_name}" not found'}
    
    product = products.data[0]
    return {
        'success': True,
        'message': f'{product["name"]}: {product["quantity"]} units in stock',
        'product': product
    }

def update_stock_from_voice(user_id, entities):
    """Update stock from voice command"""
    product_name = entities.get('product_name')
    quantity = entities.get('quantity')
    
    if not product_name or quantity is None:
        return {'success': False, 'message': 'Product name and quantity required'}
    
    products = supabase.table('products').select('*').eq('user_id', user_id).ilike('name', f'%{product_name}%').execute()
    if not products.data:
        return {'success': False, 'message': f'Product "{product_name}" not found'}
    
    product = products.data[0]
    result = supabase.table('products').update({'quantity': quantity}).eq('id', product['id']).execute()
    
    if result.data:
        return {'success': True, 'message': f'Updated {product["name"]} stock to {quantity} units'}
    return {'success': False, 'message': 'Failed to update stock'}

# Inventory Management Routes
@app.route('/api/inventory/products', methods=['GET'])
@token_required
def get_products(current_user_id):
    try:
        result = supabase.table('products').select('*').eq('user_id', current_user_id).order('created_at.desc').execute()
        return jsonify({'products': result.data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/products', methods=['POST'])
@token_required
def add_product(current_user_id):
    try:
        data = request.get_json()
        
        product_data = {
            'user_id': current_user_id,
            'name': data.get('name'),
            'description': data.get('description', ''),
            'sku': data.get('sku', ''),
            'category': data.get('category', 'General'),
            'unit_price': data.get('unit_price', 0),
            'quantity': data.get('quantity', 0),
            'min_stock_level': data.get('min_stock_level', 0),
            'supplier': data.get('supplier', '')
        }
        
        result = supabase.table('products').insert(product_data).execute()
        
        if result.data:
            return jsonify({'message': 'Product added successfully', 'product': result.data[0]}), 201
        return jsonify({'error': 'Failed to add product'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/products/<product_id>', methods=['PUT'])
@token_required
def update_product(current_user_id, product_id):
    try:
        data = request.get_json()
        
        # Verify product belongs to user
        existing = supabase.table('products').select('id').eq('id', product_id).eq('user_id', current_user_id).execute()
        if not existing.data:
            return jsonify({'error': 'Product not found'}), 404
        
        update_data = {}
        allowed_fields = ['name', 'description', 'sku', 'category', 'unit_price', 'quantity', 'min_stock_level', 'supplier']
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase.table('products').update(update_data).eq('id', product_id).eq('user_id', current_user_id).execute()
        
        if result.data:
            return jsonify({'message': 'Product updated successfully', 'product': result.data[0]}), 200
        return jsonify({'error': 'Failed to update product'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/low-stock', methods=['GET'])
@token_required
def get_low_stock(current_user_id):
    try:
        # Get products where quantity <= min_stock_level
        result = supabase.table('products').select('*').eq('user_id', current_user_id).filter('quantity', 'lte', 'min_stock_level').execute()
        return jsonify({'low_stock_products': result.data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Financial Transaction Routes
@app.route('/api/transactions', methods=['GET'])
@token_required
def get_transactions(current_user_id):
    try:
        transaction_type = request.args.get('type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = supabase.table('transactions').select('*').eq('user_id', current_user_id)
        
        if transaction_type:
            query = query.eq('transaction_type', transaction_type)
        if start_date:
            query = query.gte('date', start_date)
        if end_date:
            query = query.lte('date', end_date)
        
        result = query.order('date.desc').execute()
        return jsonify({'transactions': result.data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transactions/sale', methods=['POST'])
@token_required
def record_sale(current_user_id):
    try:
        data = request.get_json()
        
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)
        unit_price = data.get('unit_price')
        customer_name = data.get('customer_name', 'Walk-in Customer')
        
        # Get product
        product_result = supabase.table('products').select('*').eq('id', product_id).eq('user_id', current_user_id).execute()
        if not product_result.data:
            return jsonify({'error': 'Product not found'}), 404
        
        product = product_result.data[0]
        
        # Check stock
        if product['quantity'] < quantity:
            return jsonify({'error': f'Insufficient stock. Available: {product["quantity"]}'}), 400
        
        # Use product price if not provided
        if not unit_price:
            unit_price = product['unit_price']
        
        total_amount = quantity * unit_price
        
        # Record transaction
        transaction_data = {
            'user_id': current_user_id,
            'transaction_type': 'sale',
            'amount': total_amount,
            'description': f'Sale of {quantity} {product["name"]}',
            'category': 'Sales'
        }
        transaction_result = supabase.table('transactions').insert(transaction_data).execute()
        
        if transaction_result.data:
            transaction_id = transaction_result.data[0]['id']
            
            # Record sale
            sale_data = {
                'user_id': current_user_id,
                'transaction_id': transaction_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'customer_name': customer_name
            }
            sale_result = supabase.table('sales').insert(sale_data).execute()
            
            # Update stock
            new_quantity = product['quantity'] - quantity
            supabase.table('products').update({'quantity': new_quantity}).eq('id', product_id).execute()
            
            return jsonify({
                'message': 'Sale recorded successfully', 
                'sale': sale_result.data[0] if sale_result.data else None,
                'transaction': transaction_result.data[0]
            }), 201
        
        return jsonify({'error': 'Failed to record sale'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transactions/expense', methods=['POST'])
@token_required
def record_expense(current_user_id):
    try:
        data = request.get_json()
        
        transaction_data = {
            'user_id': current_user_id,
            'transaction_type': 'expense',
            'amount': data.get('amount'),
            'description': data.get('description', ''),
            'category': data.get('category', 'General Expense')
        }
        
        result = supabase.table('transactions').insert(transaction_data).execute()
        
        if result.data:
            return jsonify({'message': 'Expense recorded successfully', 'transaction': result.data[0]}), 201
        return jsonify({'error': 'Failed to record expense'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/profit', methods=['GET'])
@token_required
def calculate_profit(current_user_id):
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Get sales
        sales_query = supabase.table('transactions').select('amount').eq('user_id', current_user_id).eq('transaction_type', 'sale')
        if start_date:
            sales_query = sales_query.gte('date', start_date)
        if end_date:
            sales_query = sales_query.lte('date', end_date)
        
        sales_result = sales_query.execute()
        total_sales = sum(transaction['amount'] for transaction in sales_result.data)
        
        # Get expenses
        expenses_query = supabase.table('transactions').select('amount').eq('user_id', current_user_id).eq('transaction_type', 'expense')
        if start_date:
            expenses_query = expenses_query.gte('date', start_date)
        if end_date:
            expenses_query = expenses_query.lte('date', end_date)
        
        expenses_result = expenses_query.execute()
        total_expenses = sum(transaction['amount'] for transaction in expenses_result.data)
        
        profit = total_sales - total_expenses
        
        return jsonify({
            'total_sales': total_sales,
            'total_expenses': total_expenses,
            'profit': profit,
            'period': {
                'start_date': start_date,
                'end_date': end_date
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/dashboard', methods=['GET'])
@token_required
def get_dashboard_data(current_user_id):
    try:
        # Get total products
        products_count = supabase.table('products').select('id', count='exact').eq('user_id', current_user_id).execute()
        
        # Get low stock count
        low_stock_count = supabase.table('products').select('id', count='exact').eq('user_id', current_user_id).filter('quantity', 'lte', 'min_stock_level').execute()
        
        # Get today's sales
        today = datetime.utcnow().date().isoformat()
        today_sales = supabase.table('transactions').select('amount').eq('user_id', current_user_id).eq('transaction_type', 'sale').gte('date', today).execute()
        total_today_sales = sum(transaction['amount'] for transaction in today_sales.data)
        
        # Get this month's sales
        month_start = datetime.utcnow().replace(day=1).date().isoformat()
        month_sales = supabase.table('transactions').select('amount').eq('user_id', current_user_id).eq('transaction_type', 'sale').gte('date', month_start).execute()
        total_month_sales = sum(transaction['amount'] for transaction in month_sales.data)
        
        # Get this month's expenses
        month_expenses = supabase.table('transactions').select('amount').eq('user_id', current_user_id).eq('transaction_type', 'expense').gte('date', month_start).execute()
        total_month_expenses = sum(transaction['amount'] for transaction in month_expenses.data)
        
        # Get recent transactions
        recent_transactions = supabase.table('transactions').select('*').eq('user_id', current_user_id).order('date.desc').limit(10).execute()
        
        return jsonify({
            'total_products': products_count.count if products_count.count else 0,
            'low_stock_items': low_stock_count.count if low_stock_count.count else 0,
            'today_sales': total_today_sales,
            'month_sales': total_month_sales,
            'month_expenses': total_month_expenses,
            'month_profit': total_month_sales - total_month_expenses,
            'recent_transactions': recent_transactions.data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Sales Management Routes
@app.route('/api/sales', methods=['GET'])
@token_required
def get_sales(current_user_id):
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = supabase.table('sales').select('''
            *,
            products(name, category),
            transactions(date)
        ''').eq('user_id', current_user_id)
        
        if start_date:
            query = query.gte('created_at', start_date)
        if end_date:
            query = query.lte('created_at', end_date)
        
        result = query.order('created_at.desc').execute()
        return jsonify({'sales': result.data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales/summary', methods=['GET'])
@token_required
def get_sales_summary(current_user_id):
    try:
        period = request.args.get('period', 'week')  # week, month, year
        
        if period == 'week':
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        elif period == 'month':
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        elif period == 'year':
            start_date = (datetime.utcnow() - timedelta(days=365)).isoformat()
        else:
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        # Get sales data
        sales_result = supabase.table('sales').select('''
            total_amount,
            quantity,
            created_at,
            products(name, category)
        ''').eq('user_id', current_user_id).gte('created_at', start_date).execute()
        
        # Calculate summary
        total_revenue = sum(sale['total_amount'] for sale in sales_result.data)
        total_items_sold = sum(sale['quantity'] for sale in sales_result.data)
        total_transactions = len(sales_result.data)
        
        # Top selling products
        product_sales = {}
        for sale in sales_result.data:
            if sale['products']:
                product_name = sale['products']['name']
                if product_name in product_sales:
                    product_sales[product_name]['quantity'] += sale['quantity']
                    product_sales[product_name]['revenue'] += sale['total_amount']
                else:
                    product_sales[product_name] = {
                        'quantity': sale['quantity'],
                        'revenue': sale['total_amount'],
                        'category': sale['products']['category']
                    }
        
        # Sort by quantity sold
        top_products = sorted(product_sales.items(), key=lambda x: x[1]['quantity'], reverse=True)[:5]
        
        return jsonify({
            'period': period,
            'summary': {
                'total_revenue': total_revenue,
                'total_items_sold': total_items_sold,
                'total_transactions': total_transactions,
                'average_transaction_value': total_revenue / total_transactions if total_transactions > 0 else 0
            },
            'top_products': [{'name': name, **data} for name, data in top_products]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Voice Commands History
@app.route('/api/voice/history', methods=['GET'])
@token_required
def get_voice_history(current_user_id):
    try:
        limit = int(request.args.get('limit', 50))
        result = supabase.table('voice_commands').select('*').eq('user_id', current_user_id).order('created_at.desc').limit(limit).execute()
        return jsonify({'voice_commands': result.data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error Correction Route
@app.route('/api/voice/correct', methods=['POST'])
@token_required
def correct_voice_command(current_user_id):
    try:
        data = request.get_json()
        command_id = data.get('command_id')
        corrected_text = data.get('corrected_text')
        
        if not command_id or not corrected_text:
            return jsonify({'error': 'Command ID and corrected text are required'}), 400
        
        # Get original command
        original_result = supabase.table('voice_commands').select('*').eq('id', command_id).eq('user_id', current_user_id).execute()
        if not original_result.data:
            return jsonify({'error': 'Voice command not found'}), 404
        
        # Process corrected command
        processed_command = voice_processor.process_command_with_gemini(corrected_text)
        
        # Update the command record
        update_data = {
            'original_text': corrected_text,
            'processed_command': json.dumps(processed_command),
            'confidence_score': processed_command.get('confidence', 0.0),
            'action_taken': 'corrected'
        }
        
        supabase.table('voice_commands').update(update_data).eq('id', command_id).execute()
        
        # Execute the corrected command
        action_result = execute_voice_command(current_user_id, processed_command)
        
        return jsonify({
            'message': 'Command corrected and executed',
            'processed_command': processed_command,
            'action_result': action_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Category Management
@app.route('/api/categories', methods=['GET'])
@token_required
def get_categories(current_user_id):
    try:
        # Get unique categories from products
        result = supabase.table('products').select('category').eq('user_id', current_user_id).execute()
        categories = list(set(product['category'] for product in result.data if product['category']))
        return jsonify({'categories': categories}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Bulk Operations
@app.route('/api/inventory/bulk-update', methods=['POST'])
@token_required
def bulk_update_inventory(current_user_id):
    try:
        data = request.get_json()
        updates = data.get('updates', [])
        
        results = []
        for update in updates:
            product_id = update.get('product_id')
            new_quantity = update.get('quantity')
            
            if product_id and new_quantity is not None:
                result = supabase.table('products').update({
                    'quantity': new_quantity,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', product_id).eq('user_id', current_user_id).execute()
                
                results.append({
                    'product_id': product_id,
                    'success': bool(result.data),
                    'new_quantity': new_quantity
                })
        
        return jsonify({
            'message': f'Bulk update completed. {len([r for r in results if r["success"]])} products updated.',
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Export Data
@app.route('/api/export/transactions', methods=['GET'])
@token_required
def export_transactions(current_user_id):
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        format_type = request.args.get('format', 'json')  # json or csv
        
        query = supabase.table('transactions').select('*').eq('user_id', current_user_id)
        
        if start_date:
            query = query.gte('date', start_date)
        if end_date:
            query = query.lte('date', end_date)
        
        result = query.order('date.desc').execute()
        
        if format_type == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['id', 'transaction_type', 'amount', 'description', 'category', 'date'])
            writer.writeheader()
            
            for transaction in result.data:
                writer.writerow({
                    'id': transaction['id'],
                    'transaction_type': transaction['transaction_type'],
                    'amount': transaction['amount'],
                    'description': transaction['description'],
                    'category': transaction['category'],
                    'date': transaction['date']
                })
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'transactions_{datetime.utcnow().strftime("%Y%m%d")}.csv'
            )
        else:
            return jsonify({'transactions': result.data}), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

# User Profile Management
@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user_id):
    try:
        result = supabase.table('users').select('id, email, first_name, last_name, business_name, phone, created_at').eq('id', current_user_id).execute()
        if result.data:
            return jsonify({'user': result.data[0]}), 200
        return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/profile', methods=['PUT'])
@token_required
def update_user_profile(current_user_id):
    try:
        data = request.get_json()
        
        update_data = {}
        allowed_fields = ['first_name', 'last_name', 'business_name', 'phone']
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        if update_data:
            update_data['updated_at'] = datetime.utcnow().isoformat()
            result = supabase.table('users').update(update_data).eq('id', current_user_id).execute()
            
            if result.data:
                return jsonify({'message': 'Profile updated successfully', 'user': result.data[0]}), 200
        
        return jsonify({'error': 'No valid fields to update'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize database tables (call this once)
@app.route('/api/init-db', methods=['POST'])
def initialize_database():
    try:
        init_db()
        return jsonify({'message': 'Database initialized successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)