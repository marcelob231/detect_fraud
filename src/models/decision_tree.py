from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """
    Implementação específica do modelo Decision Tree para detecção de fraude.
    """
    
    def __init__(self, **kwargs):
        # Definir hiperparâmetros padrão
        default_params = self.get_default_params()
        default_params.update(kwargs)
        
        super().__init__(model_name="Decision Tree", **default_params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna hiperparâmetros padrão otimizados para detecção de fraude."""
        return {
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 100,
            'min_samples_leaf': 50,
            'max_features': 'sqrt',
            'class_weight': 'balanced',  # Para lidar com desbalanceamento se necessário
            'criterion': 'gini'
        }
    
    def create_model(self) -> DecisionTreeClassifier:
        """Cria e retorna o modelo DecisionTreeClassifier."""
        return DecisionTreeClassifier(**self.hyperparameters)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Treina o modelo Decision Tree e retorna métricas detalhadas.
        """
        # Executar treinamento da classe base
        training_metadata = super().train(X_train, y_train, X_val, y_val)
        
        # Adicionar métricas específicas do Decision Tree
        training_metadata.update({
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves(),
            'n_features_used': self.model.n_features_in_
        })
        
        print(f"🌳 Árvore criada:")
        print(f"   Profundidade: {training_metadata['tree_depth']}")
        print(f"   Folhas: {training_metadata['n_leaves']}")
        print(f"   Features utilizadas: {training_metadata['n_features_used']}")
        
        return training_metadata
    
    def get_detailed_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia o modelo com métricas detalhadas para detecção de fraude.
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes da avaliação!")
        
        # Fazer predições
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calcular métricas básicas
        accuracy = (predictions == y_test).mean()
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, predictions)
        
        # Relatório de classificação
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Métricas específicas para fraude (classe 1)
        fraud_metrics = report['1'] if '1' in report else {}
        normal_metrics = report['0'] if '0' in report else {}
        
        evaluation_results = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'fraud_precision': fraud_metrics.get('precision', 0),
            'fraud_recall': fraud_metrics.get('recall', 0),
            'fraud_f1': fraud_metrics.get('f1-score', 0),
            'normal_precision': normal_metrics.get('precision', 0),
            'normal_recall': normal_metrics.get('recall', 0),
            'normal_f1': normal_metrics.get('f1-score', 0),
            'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0),
            'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0),
            'n_test_samples': len(y_test),
            'n_fraud_samples': int(y_test.sum()),
            'n_normal_samples': int(len(y_test) - y_test.sum())
        }
        
        # Imprimir resultados
        print("🎯 AVALIAÇÃO DO MODELO:")
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"   Precisão (Fraude): {fraud_metrics.get('precision', 0):.4f}")
        print(f"   Recall (Fraude): {fraud_metrics.get('recall', 0):.4f}")
        print(f"   F1-Score (Fraude): {fraud_metrics.get('f1-score', 0):.4f}")
        print(f"   F1-Score Macro: {report.get('macro avg', {}).get('f1-score', 0):.4f}")
        
        return evaluation_results
    
    def get_top_features(self, feature_names: list, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna as features mais importantes do modelo.
        """
        importance = self.get_feature_importance()
        if importance is None:
            return pd.DataFrame()
        
        importance.index = feature_names
        top_features = importance.nlargest(top_n).reset_index()
        top_features.columns = ['feature', 'importance']
        
        print(f"🏆 TOP {top_n} FEATURES MAIS IMPORTANTES:")
        for i, row in top_features.iterrows():
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return top_features
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 15, 
                              figsize: tuple = (10, 8), save_path: Optional[str] = None):
        """
        Plota a importância das features.
        """
        importance = self.get_feature_importance()
        if importance is None:
            print("❌ Modelo não suporta importância de features")
            return None
        
        # Preparar dados para o plot
        importance.index = feature_names
        top_importance = importance.nlargest(top_n)
        
        # Criar o plot
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_importance)), top_importance.values)
        plt.yticks(range(len(top_importance)), top_importance.index)
        plt.xlabel('Importância')
        plt.title(f'Top {top_n} Features Mais Importantes - Decision Tree')
        plt.gca().invert_yaxis()
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico salvo em: {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def plot_tree_structure(self, feature_names: list, max_depth: int = 3, 
                           figsize: tuple = (20, 10), save_path: Optional[str] = None):
        """
        Plota a estrutura da árvore de decisão (apenas primeiros níveis).
        """
        if not self.is_trained:
            print("❌ Modelo deve ser treinado antes de plotar a árvore")
            return None
        
        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                 feature_names=feature_names,
                 class_names=['Normal', 'Fraude'],
                 filled=True,
                 rounded=True,
                 max_depth=max_depth,
                 fontsize=10)
        
        plt.title(f'Estrutura da Árvore de Decisão (profundidade máxima: {max_depth})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🌳 Árvore salva em: {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def get_decision_path(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna o caminho de decisão para uma amostra específica.
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado antes de analisar o caminho de decisão")
        
        # Obter o caminho de decisão
        leaf_id = self.model.decision_path(X_sample).toarray()
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        
        decision_info = {
            'prediction': self.predict(X_sample)[0],
            'probability': self.predict_proba(X_sample)[0].tolist(),
            'path_length': leaf_id.sum(),
            'decision_path': []
        }
        
        # Construir o caminho de decisão
        sample_id = 0
        node_indicator = leaf_id[sample_id, :]
        
        for node_id in range(len(node_indicator)):
            if node_indicator[node_id]:
                if X_sample.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                    direction = "≤"
                else:
                    direction = ">"
                
                decision_info['decision_path'].append({
                    'node': node_id,
                    'feature': feature[node_id],
                    'threshold': threshold[node_id],
                    'direction': direction,
                    'value': X_sample.iloc[sample_id, feature[node_id]]
                })
        
        return decision_info


# Função de conveniência para uso rápido
def create_and_train_decision_tree(train_data_path: str = "data/processed/decision_tree/train.csv",
                                  val_data_path: str = "data/processed/decision_tree/validation.csv",
                                  **model_params) -> DecisionTreeModel:
    """
    Função de conveniência para criar e treinar um Decision Tree rapidamente.
    """
    # Carregar dados
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    
    # Separar features e target
    X_train = train_df.drop(['Class'], axis=1)
    y_train = train_df['Class']
    X_val = val_df.drop(['Class'], axis=1) 
    y_val = val_df['Class']
    
    # Criar e treinar modelo
    model = DecisionTreeModel(**model_params)
    model.train(X_train, y_train, X_val, y_val)
    
    return model


if __name__ == "__main__":
    # Teste rápido
    print("🌳 Testando Decision Tree...")
    model = create_and_train_decision_tree()
    print("✅ Decision Tree criado e treinado com sucesso!")