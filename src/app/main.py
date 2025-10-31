from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from datetime import timedelta
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configurações
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'fraud-detection-secret-key-2025')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    app.config['MODEL_STORAGE_PATH'] = 'results/'
    app.config['DATA_PATH'] = 'data/processed/'
    
    # Configurar CORS para Angular
    CORS(app, origins=[
        'http://localhost:4200',  # Angular dev server (porta padrão)
        'http://localhost:4300',  # Angular dev server (porta atual)
        'http://localhost:3000',  # Backup
        'http://127.0.0.1:4200',
        'http://127.0.0.1:4300'
    ], supports_credentials=True)
    
    # Configurar JWT
    jwt = JWTManager(app)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Rota principal
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Fraud Detection API',
            'version': '1.0.0',
            'status': 'running'
        })
    
    # Rota de health check
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'message': 'Fraud Detection API is running',
            'available_models': ['decision_tree'],
            'api_version': '1.0.0'
        })
    
    # Registrar blueprints (rotas)
    try:
        # Importar rotas usando imports absolutos
        import sys
        from pathlib import Path
        
        # Adicionar o diretório src ao path
        current_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(current_dir / 'src'))
        
        from app.routes.auth_routes import auth_bp
        from app.routes.models_routes import models_bp
        from app.routes.visualization_routes import visualization_bp
        
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        app.register_blueprint(models_bp, url_prefix='/api/models')
        app.register_blueprint(visualization_bp, url_prefix='/api/visualization')
        logger.info('All routes registered successfully')
    except ImportError as e:
        logger.warning(f'Some routes not available yet: {e}')
    
    # Handlers de erro
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Internal error: {error}')
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required'
        }), 401
    
    # JWT error handlers
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token expired',
            'message': 'The JWT token has expired'
        }), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'error': 'Invalid token',
            'message': 'The JWT token is invalid'
        }), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'error': 'Authorization required',
            'message': 'JWT token is required'
        }), 401
    
    logger.info('Fraud Detection API initialized successfully')
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)