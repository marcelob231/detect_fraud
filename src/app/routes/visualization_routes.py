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

visualization_bp = Blueprint('visualization', __name__)
logger = logging.getLogger(__name__)

@visualization_bp.route('/decision_tree/confusion-matrix', methods=['GET'])
@jwt_required()
def get_confusion_matrix():
    """
    Retorna dados da matriz de confusão formatados para gráficos Angular.
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
        cm = evaluation['confusion_matrix']
        
        # Preparar dados para heatmap Angular
        heatmap_data = {
            'data': [
                {'x': 'Normal', 'y': 'Normal', 'value': cm[0][0], 'label': 'Verdadeiro Negativo'},
                {'x': 'Fraude', 'y': 'Normal', 'value': cm[0][1], 'label': 'Falso Positivo'},
                {'x': 'Normal', 'y': 'Fraude', 'value': cm[1][0], 'label': 'Falso Negativo'},
                {'x': 'Fraude', 'y': 'Fraude', 'value': cm[1][1], 'label': 'Verdadeiro Positivo'}
            ],
            'labels': ['Normal', 'Fraude'],
            'title': 'Matriz de Confusão - Decision Tree'
        }
        
        # Calcular percentuais
        total = sum(sum(row) for row in cm)
        percentages = {
            'true_negative': round((cm[0][0] / total) * 100, 2),
            'false_positive': round((cm[0][1] / total) * 100, 2),
            'false_negative': round((cm[1][0] / total) * 100, 2),
            'true_positive': round((cm[1][1] / total) * 100, 2)
        }
        
        # Métricas resumidas
        metrics = {
            'accuracy': evaluation['accuracy'],
            'precision': evaluation['fraud_precision'],
            'recall': evaluation['fraud_recall'],
            'f1_score': evaluation['fraud_f1']
        }
        
        return jsonify({
            'heatmap_data': heatmap_data,
            'raw_matrix': cm,
            'percentages': percentages,
            'metrics': metrics,
            'total_samples': total,
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Confusion matrix error: {str(e)}')
        return jsonify({
            'error': 'Failed to generate confusion matrix',
            'message': str(e)
        }), 500

@visualization_bp.route('/decision_tree/probability-distribution', methods=['GET'])
@jwt_required()
def get_probability_distribution():
    """
    Retorna distribuição das probabilidades de predição para histogramas.
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
        
        # Fazer predições
        probabilities = model.predict_proba(X_test)
        fraud_probs = probabilities[:, 1]  # Probabilidades de fraude
        
        # Separar por classe real
        normal_probs = fraud_probs[y_test == 0].tolist()
        fraud_probs_real = fraud_probs[y_test == 1].tolist()
        
        # Preparar dados para histograma Angular
        histogram_data = {
            'normal_transactions': {
                'data': normal_probs,
                'label': 'Transações Normais',
                'color': '#3498db',
                'bins': 50
            },
            'fraud_transactions': {
                'data': fraud_probs_real,
                'label': 'Transações Fraudulentas',
                'color': '#e74c3c',
                'bins': 50
            }
        }
        
        # Estatísticas
        stats = {
            'normal': {
                'mean': float(np.mean(normal_probs)),
                'median': float(np.median(normal_probs)),
                'std': float(np.std(normal_probs)),
                'min': float(np.min(normal_probs)),
                'max': float(np.max(normal_probs)),
                'count': len(normal_probs)
            },
            'fraud': {
                'mean': float(np.mean(fraud_probs_real)),
                'median': float(np.median(fraud_probs_real)),
                'std': float(np.std(fraud_probs_real)),
                'min': float(np.min(fraud_probs_real)),
                'max': float(np.max(fraud_probs_real)),
                'count': len(fraud_probs_real)
            }
        }
        
        return jsonify({
            'histogram_data': histogram_data,
            'statistics': stats,
            'title': 'Distribuição das Probabilidades de Fraude',
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Probability distribution error: {str(e)}')
        return jsonify({
            'error': 'Failed to generate probability distribution',
            'message': str(e)
        }), 500

@visualization_bp.route('/decision_tree/feature-distribution', methods=['GET'])
@jwt_required()
def get_feature_distribution():
    """
    Retorna distribuição das top features por classe para visualização.
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
        
        # Obter top 6 features mais importantes
        importance = model.get_feature_importance()
        feature_names = X_test.columns.tolist()
        importance.index = feature_names
        top_6_features = importance.nlargest(6).index.tolist()
        
        # Preparar dados para cada feature
        features_data = {}
        
        for feature in top_6_features:
            normal_data = X_test[y_test == 0][feature].tolist()
            fraud_data = X_test[y_test == 1][feature].tolist()
            
            features_data[feature] = {
                'importance': float(importance[feature]),
                'normal_distribution': {
                    'data': normal_data,
                    'mean': float(np.mean(normal_data)),
                    'std': float(np.std(normal_data)),
                    'label': 'Normal'
                },
                'fraud_distribution': {
                    'data': fraud_data,
                    'mean': float(np.mean(fraud_data)),
                    'std': float(np.std(fraud_data)),
                    'label': 'Fraude'
                }
            }
        
        return jsonify({
            'features_data': features_data,
            'top_features': top_6_features,
            'title': 'Distribuição das Top Features por Classe',
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Feature distribution error: {str(e)}')
        return jsonify({
            'error': 'Failed to generate feature distribution',
            'message': str(e)
        }), 500

@visualization_bp.route('/decision_tree/performance-metrics', methods=['GET'])
@jwt_required()
def get_performance_metrics():
    """
    Retorna métricas de performance organizadas para dashboards.
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
        
        # Organizar métricas para cards/widgets
        performance_cards = [
            {
                'title': 'Acurácia',
                'value': float(evaluation['accuracy']),
                'percentage': round(float(evaluation['accuracy']) * 100, 2),
                'icon': 'accuracy',
                'color': 'primary',
                'description': 'Percentual de predições corretas'
            },
            {
                'title': 'Precisão (Fraude)',
                'value': float(evaluation['fraud_precision']),
                'percentage': round(float(evaluation['fraud_precision']) * 100, 2),
                'icon': 'precision',
                'color': 'success',
                'description': 'Das predições de fraude, quantas estavam corretas'
            },
            {
                'title': 'Recall (Fraude)',
                'value': float(evaluation['fraud_recall']),
                'percentage': round(float(evaluation['fraud_recall']) * 100, 2),
                'icon': 'recall',
                'color': 'warning',
                'description': 'Das fraudes reais, quantas foram detectadas'
            },
            {
                'title': 'F1-Score',
                'value': float(evaluation['fraud_f1']),
                'percentage': round(float(evaluation['fraud_f1']) * 100, 2),
                'icon': 'f1score',
                'color': 'info',
                'description': 'Média harmônica entre precisão e recall'
            }
        ]
        
        # Dados para gráfico de radar/spider
        radar_data = {
            'labels': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
            'datasets': [{
                'label': 'Decision Tree',
                'data': [
                    float(evaluation['accuracy']),
                    float(evaluation['fraud_precision']),
                    float(evaluation['fraud_recall']),
                    float(evaluation['fraud_f1'])
                ],
                'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 2
            }]
        }
        
        # Informações do modelo
        model_info = {
            'training_samples': int(model.training_metadata.get('training_samples', 0)),
            'tree_depth': int(model.training_metadata.get('tree_depth', 0)),
            'n_leaves': int(model.training_metadata.get('n_leaves', 0)),
            'training_time': float(model.training_metadata.get('training_time_seconds', 0)),
            'features_count': int(model.training_metadata.get('n_features_used', 0))
        }
        
        return jsonify({
            'performance_cards': performance_cards,
            'radar_chart': radar_data,
            'model_info': model_info,
            'detailed_metrics': evaluation,
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Performance metrics error: {str(e)}')
        return jsonify({
            'error': 'Failed to generate performance metrics',
            'message': str(e)
        }), 500

@visualization_bp.route('/decision_tree/tree-structure', methods=['GET'])
@jwt_required()
def get_tree_structure():
    """
    Retorna dados da estrutura da árvore de decisão para visualização hierárquica.
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
        
        # Obter estrutura da árvore
        tree = model.model.tree_
        
        # Carregar dados de teste para obter nomes das features
        test_df = pd.read_csv('data/processed/decision_tree/test.csv')
        X_test = test_df.drop(['Class'], axis=1)
        feature_names = X_test.columns.tolist()
        
        def build_tree_node(node_id, depth=0, max_depth=4):
            """Constrói recursivamente a estrutura da árvore."""
            if depth > max_depth or node_id == -1:
                return None
            
            # Informações do nó
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            samples = tree.n_node_samples[node_id]
            value = tree.value[node_id][0]  # [normal_count, fraud_count]
            
            # Se é folha (nó terminal)
            is_leaf = feature_idx == -2
            
            if is_leaf:
                # Nó folha - determinar classe predita
                predicted_class = 'Normal' if value[0] > value[1] else 'Fraude'
                node_data = {
                    'id': f'leaf_{node_id}',
                    'type': 'leaf',
                    'predicted_class': predicted_class,
                    'samples': int(samples),
                    'distribution': {
                        'normal': int(value[0]),
                        'fraud': int(value[1])
                    },
                    'purity': max(value) / sum(value),
                    'depth': depth
                }
            else:
                # Nó interno - tem condição de split
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                
                node_data = {
                    'id': f'node_{node_id}',
                    'type': 'internal',
                    'feature': feature_name,
                    'threshold': float(threshold),
                    'condition': f'{feature_name} <= {threshold:.4f}',
                    'samples': int(samples),
                    'distribution': {
                        'normal': int(value[0]),
                        'fraud': int(value[1])
                    },
                    'depth': depth,
                    'children': []
                }
                
                # Adicionar filhos recursivamente
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
                
                if left_child != -1:
                    left_node = build_tree_node(left_child, depth + 1, max_depth)
                    if left_node:
                        left_node['parent_condition'] = 'True'
                        node_data['children'].append(left_node)
                
                if right_child != -1:
                    right_node = build_tree_node(right_child, depth + 1, max_depth)
                    if right_node:
                        right_node['parent_condition'] = 'False'
                        node_data['children'].append(right_node)
            
            return node_data
        
        # Construir árvore a partir da raiz
        tree_structure = build_tree_node(0)
        
        # Estatísticas da árvore completa
        tree_stats = {
            'total_depth': int(tree.max_depth),
            'total_nodes': int(tree.node_count),
            'total_leaves': int(np.sum(tree.children_left == -1)),
            'features_used': int(model.training_metadata.get('n_features_used', 0)),
            'max_display_depth': 4
        }
        
        return jsonify({
            'tree_structure': tree_structure,
            'tree_statistics': tree_stats,
            'feature_names': feature_names,
            'class_names': ['Normal', 'Fraude'],
            'title': 'Estrutura da Árvore de Decisão',
            'model_id': 'decision_tree',
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Tree structure error: {str(e)}')
        return jsonify({
            'error': 'Failed to generate tree structure',
            'message': str(e)
        }), 500