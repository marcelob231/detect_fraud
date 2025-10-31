from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import check_password_hash, generate_password_hash
import logging

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

# Usuários mockados para demonstração (em produção, usar banco de dados)
USERS = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin',
        'name': 'Administrator'
    },
    'analyst': {
        'password': generate_password_hash('analyst123'),
        'role': 'analyst', 
        'name': 'Data Analyst'
    },
    'viewer': {
        'password': generate_password_hash('viewer123'),
        'role': 'viewer',
        'name': 'Report Viewer'
    }
}

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Autenticação do usuário.
    Retorna JWT token para acesso às APIs.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'JSON data required'
            }), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'error': 'Missing credentials',
                'message': 'Username and password are required'
            }), 400
        
        # Verificar usuário
        user = USERS.get(username)
        if not user or not check_password_hash(user['password'], password):
            logger.warning(f'Failed login attempt for username: {username}')
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401
        
        # Criar token JWT
        additional_claims = {
            'role': user['role'],
            'name': user['name']
        }
        
        access_token = create_access_token(
            identity=username,
            additional_claims=additional_claims
        )
        
        logger.info(f'Successful login for user: {username}')
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'username': username,
                'name': user['name'],
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        logger.error(f'Login error: {str(e)}')
        return jsonify({
            'error': 'Login failed',
            'message': 'An error occurred during login'
        }), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Retorna informações do usuário logado.
    Requer token JWT válido.
    """
    try:
        current_user = get_jwt_identity()
        user_data = USERS.get(current_user)
        
        if not user_data:
            return jsonify({
                'error': 'User not found',
                'message': 'User data not available'
            }), 404
        
        return jsonify({
            'user': {
                'username': current_user,
                'name': user_data['name'],
                'role': user_data['role']
            }
        }), 200
        
    except Exception as e:
        logger.error(f'Profile error: {str(e)}')
        return jsonify({
            'error': 'Profile fetch failed',
            'message': 'Could not retrieve user profile'
        }), 500

@auth_bp.route('/verify', methods=['GET'])
@jwt_required()
def verify_token():
    """
    Verifica se o token JWT é válido.
    Usado pelo frontend para validar sessão.
    """
    try:
        current_user = get_jwt_identity()
        return jsonify({
            'valid': True,
            'user': current_user,
            'message': 'Token is valid'
        }), 200
        
    except Exception as e:
        logger.error(f'Token verification error: {str(e)}')
        return jsonify({
            'valid': False,
            'message': 'Token verification failed'
        }), 401

@auth_bp.route('/users', methods=['GET'])
@jwt_required()
def list_users():
    """
    Lista usuários disponíveis (apenas para demonstração).
    Em produção, restringir por role.
    """
    try:
        users_list = []
        for username, user_data in USERS.items():
            users_list.append({
                'username': username,
                'name': user_data['name'],
                'role': user_data['role']
            })
        
        return jsonify({
            'users': users_list,
            'total': len(users_list)
        }), 200
        
    except Exception as e:
        logger.error(f'List users error: {str(e)}')
        return jsonify({
            'error': 'Failed to list users',
            'message': 'Could not retrieve users list'
        }), 500

# Endpoint de informações sobre autenticação
@auth_bp.route('/info', methods=['GET']) 
def auth_info():
    """
    Informações sobre o sistema de autenticação.
    Endpoint público para o frontend conhecer os requisitos.
    """
    return jsonify({
        'authentication': {
            'type': 'JWT',
            'header': 'Authorization',
            'prefix': 'Bearer',
            'expiration': '24 hours'
        },
        'demo_users': [
            {
                'username': 'admin',
                'password': 'admin123',
                'role': 'admin',
                'permissions': ['full_access', 'train_models', 'view_results']
            },
            {
                'username': 'analyst', 
                'password': 'analyst123',
                'role': 'analyst',
                'permissions': ['view_results', 'run_analysis']
            },
            {
                'username': 'viewer',
                'password': 'viewer123', 
                'role': 'viewer',
                'permissions': ['view_results']
            }
        ],
        'message': 'Use these demo credentials for testing'
    }), 200