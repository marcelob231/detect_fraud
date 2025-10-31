from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Adicionar src ao path para imports
current_dir = Path(__file__).parent.parent.parent.parent
src_path = str(current_dir / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

models_bp = Blueprint('models', __name__)
logger = logging.getLogger(__name__)

@models_bp.route('/', methods=['GET'])
@jwt_required()
def list_models():
    """
    Lista todos os modelos disponíveis no sistema.
    """
    try:
        models = [
            {
                'id': 'decision_tree',
                'name': 'Decision Tree',
                'description': 'Árvore de decisão para classificação de fraudes',
                'status': 'available',
                'type': 'supervised',
                'category': 'classification'
            }
            # Futuros modelos serão adicionados aqui
        ]
        
        return jsonify({
            'models': models,
            'total': len(models),
            'message': 'Models retrieved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f'List models error: {str(e)}')
        return jsonify({
            'error': 'Failed to list models',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/info', methods=['GET'])
@jwt_required()
def get_decision_tree_info():
    """
    Retorna informações detalhadas sobre o modelo Decision Tree.
    """
    try:
        # Verificar se modelo existe
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        model_trained = model_path.exists()
        
        model_info = {
            'model_id': 'decision_tree',
            'name': 'Decision Tree Classifier',
            'description': 'Modelo de árvore de decisão para detecção de fraudes em transações',
            'algorithm_type': 'Decision Tree',
            'category': 'classification',
            'model_trained': model_trained,
            'hyperparameters': {
                'max_depth': 10,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'class_weight': 'balanced'
            }
        }
        
        # Se o modelo estiver treinado, adicionar informações de treinamento
        if model_trained:
            try:
                from models.decision_tree import DecisionTreeModel
                model = DecisionTreeModel()
                model.load_model(str(model_path))
                
                model_info['training_info'] = {
                    'tree_depth': int(model.training_metadata.get('tree_depth', 0)),
                    'n_leaves': int(model.training_metadata.get('n_leaves', 0)),
                    'training_samples': int(model.training_metadata.get('training_samples', 0)),
                    'features_count': int(model.training_metadata.get('n_features_used', 0)),
                    'training_time': float(model.training_metadata.get('training_time_seconds', 0)),
                    'trained_at': str(model.training_metadata.get('trained_at', ''))
                }
            except Exception as e:
                logger.warning(f'Could not load model metadata: {e}')
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f'Model info error: {str(e)}')
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/info', methods=['GET'])
@jwt_required()
def decision_tree_info():
    """
    Informações detalhadas sobre o modelo Decision Tree.
    """
    try:
        # Verificar se existe modelo treinado
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        metadata_path = Path('data/processed/metadata/decision_tree_metadata.json')
        
        model_exists = model_path.exists()
        metadata_exists = metadata_path.exists()
        
        info = {
            'model_id': 'decision_tree',
            'name': 'Decision Tree Classifier',
            'description': 'Árvore de decisão otimizada para detecção de fraudes em cartão de crédito',
            'algorithm_type': 'Supervised Learning',
            'category': 'Classification',
            'model_trained': model_exists,
            'metadata_available': metadata_exists,
            'hyperparameters': {
                'max_depth': 10,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'random_state': 42
            }
        }
        
        # Se existir metadata, incluir informações de treinamento
        if metadata_exists:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                info['training_info'] = {
                    'total_records': metadata.get('total_records'),
                    'fraud_percentage': metadata.get('fraud_percentage'),
                    'splits': metadata.get('splits'),
                    'trained_at': metadata.get('created_at')
                }
            except Exception as e:
                logger.warning(f'Could not read metadata: {e}')
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f'Decision tree info error: {str(e)}')
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/status', methods=['GET'])
@jwt_required()
def get_decision_tree_status():
    """
    Retorna status rápido do modelo Decision Tree.
    """
    try:
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        is_trained = model_path.exists()
        
        status = {
            'model_id': 'decision_tree',
            'trained': is_trained,
            'status': 'ready' if is_trained else 'not_trained',
            'model_file': str(model_path) if is_trained else None,
            'checked_at': datetime.now().isoformat()
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f'Status check error: {str(e)}')
        return jsonify({
            'error': 'Failed to check model status',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/train', methods=['POST'])
@jwt_required()
def train_decision_tree():
    """
    Treina o modelo Decision Tree.
    Aceita parâmetros customizados no body da requisição.
    """
    try:
        current_user = get_jwt_identity()
        logger.info(f'Training request from user: {current_user}')
        
        # Parâmetros do modelo (opcional)
        data = request.get_json() or {}
        custom_params = data.get('parameters', {})
        
        # Importar e treinar modelo
        from app.load.loader import prepare_data_for_decision_tree
        from models.decision_tree import DecisionTreeModel
        
        # Preparar dados
        files, class_weights, metadata = prepare_data_for_decision_tree()
        
        # Carregar dados de treino e validação
        train_df = pd.read_csv(files['train'])
        val_df = pd.read_csv(files['validation'])
        
        X_train = train_df.drop(['Class'], axis=1)
        y_train = train_df['Class']
        X_val = val_df.drop(['Class'], axis=1)
        y_val = val_df['Class']
        
        # Criar modelo com parâmetros customizados
        model = DecisionTreeModel(**custom_params)
        
        # Treinar
        training_metadata = model.train(X_train, y_train, X_val, y_val)
        
        # Salvar modelo
        model_dir = Path('results/decision_tree_analysis')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'decision_tree_model.joblib'
        model.save_model(str(model_path))
        
        response_data = {
            'message': 'Model trained successfully',
            'model_id': 'decision_tree',
            'training_metadata': training_metadata,
            'trained_by': current_user,
            'trained_at': datetime.now().isoformat(),
            'model_saved': str(model_path)
        }
        
        logger.info(f'Decision tree trained successfully by {current_user}')
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f'Training error: {str(e)}')
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/evaluate', methods=['GET'])
@jwt_required()
def evaluate_decision_tree():
    """
    Avalia o modelo Decision Tree no conjunto de teste.
    """
    try:
        # Verificar se modelo existe
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        if not model_path.exists():
            return jsonify({
                'error': 'Model not found',
                'message': 'Decision tree model needs to be trained first'
            }), 404
        
        # Carregar modelo
        from models.decision_tree import DecisionTreeModel
        model = DecisionTreeModel()
        model.load_model(str(model_path))
        
        # Carregar dados de teste
        test_df = pd.read_csv('data/processed/decision_tree/test.csv')
        X_test = test_df.drop(['Class'], axis=1)
        y_test = test_df['Class']
        
        # Avaliar modelo
        evaluation = model.get_detailed_evaluation(X_test, y_test)
        
        # Adicionar informações extras para o frontend
        evaluation['model_id'] = 'decision_tree'
        evaluation['evaluated_at'] = datetime.now().isoformat()
        evaluation['test_samples'] = len(y_test)
        
        return jsonify({
            'evaluation': evaluation,
            'message': 'Model evaluated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f'Evaluation error: {str(e)}')
        return jsonify({
            'error': 'Evaluation failed',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/feature-importance', methods=['GET'])
@jwt_required()
def get_feature_importance():
    """
    Retorna a importância das features do Decision Tree.
    Dados formatados para gráficos no frontend Angular.
    """
    try:
        # Verificar se modelo existe
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        if not model_path.exists():
            return jsonify({
                'error': 'Model not found',
                'message': 'Decision tree model needs to be trained first'
            }), 404
        
        # Carregar modelo
        from models.decision_tree import DecisionTreeModel
        model = DecisionTreeModel()
        model.load_model(str(model_path))
        
        # Obter nomes das features
        test_df = pd.read_csv('data/processed/decision_tree/test.csv')
        feature_names = test_df.drop(['Class'], axis=1).columns.tolist()
        
        # Obter importância
        importance_series = model.get_feature_importance()
        importance_series.index = feature_names
        
        # Preparar dados para o frontend (top 15)
        top_features = importance_series.nlargest(15)
        
        chart_data = {
            'labels': top_features.index.tolist(),
            'values': top_features.values.tolist(),
            'title': 'Feature Importance - Decision Tree',
            'type': 'horizontal_bar'
        }
        
        # Dados tabulares para exibição
        table_data = []
        for i, (feature, importance) in enumerate(top_features.items()):
            table_data.append({
                'rank': i + 1,
                'feature': feature,
                'importance': round(importance, 4),
                'percentage': round(importance * 100, 2)
            })
        
        return jsonify({
            'chart_data': chart_data,
            'table_data': table_data,
            'total_features': len(feature_names),
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Feature importance error: {str(e)}')
        return jsonify({
            'error': 'Failed to get feature importance',
            'message': str(e)
        }), 500

@models_bp.route('/decision_tree/predict', methods=['POST'])
@jwt_required()
def predict_sample():
    """
    Faz predição em uma amostra específica.
    Aceita dados da transação no body da requisição.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'JSON data with transaction features required'
            }), 400
        
        # Verificar se modelo existe
        model_path = Path('results/decision_tree_analysis/decision_tree_model.joblib')
        if not model_path.exists():
            return jsonify({
                'error': 'Model not found',
                'message': 'Decision tree model needs to be trained first'
            }), 404
        
        # Carregar modelo
        from models.decision_tree import DecisionTreeModel
        model = DecisionTreeModel()
        model.load_model(str(model_path))
        
        # Preparar dados da amostra
        transaction_data = data.get('transaction')
        if not transaction_data:
            return jsonify({
                'error': 'Missing transaction data',
                'message': 'Transaction features are required'
            }), 400
        
        # Converter para DataFrame
        sample_df = pd.DataFrame([transaction_data])
        
        # Fazer predição
        prediction = model.predict(sample_df)[0]
        probabilities = model.predict_proba(sample_df)[0]
        
        result = {
            'prediction': {
                'class': int(prediction),
                'label': 'Fraude' if prediction == 1 else 'Normal',
                'confidence': float(max(probabilities))
            },
            'probabilities': {
                'normal': float(probabilities[0]),
                'fraud': float(probabilities[1])
            },
            'model_id': 'decision_tree',
            'predicted_at': datetime.now().isoformat()
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500