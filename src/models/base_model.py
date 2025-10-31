from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import json
from datetime import datetime

class BaseModel(ABC):
    """
    Classe base abstrata para todos os modelos de Machine Learning.
    Define a interface comum que todos os modelos devem implementar.
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self.hyperparameters = kwargs
        
    @abstractmethod
    def create_model(self) -> Any:
        """Cria e retorna o modelo específico (ex: DecisionTreeClassifier)."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna os hiperparâmetros padrão para o modelo."""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Treina o modelo com os dados fornecidos.
        
        Args:
            X_train: Features de treinamento
            y_train: Labels de treinamento  
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Dicionário com métricas de treinamento
        """
        print(f"🚀 Iniciando treinamento do {self.model_name}...")
        
        # Criar o modelo se ainda não existe
        if self.model is None:
            self.model = self.create_model()
        
        # Registrar início do treinamento
        start_time = datetime.now()
        
        # Treinar o modelo
        self.model.fit(X_train, y_train)
        
        # Calcular tempo de treinamento
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Avaliar no conjunto de treinamento
        train_predictions = self.model.predict(X_train)
        train_accuracy = (train_predictions == y_train).mean()
        
        # Avaliar no conjunto de validação se fornecido
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = (val_predictions == y_val).mean()
        
        # Salvar metadados do treinamento
        self.training_metadata = {
            'model_name': self.model_name,
            'training_samples': len(X_train),
            'features_count': len(X_train.columns),
            'training_time_seconds': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'hyperparameters': self.get_current_params(),
            'trained_at': start_time.isoformat()
        }
        
        self.is_trained = True
        
        print(f"✅ Treinamento concluído em {training_time:.2f}s")
        print(f"   Acurácia treino: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"   Acurácia validação: {val_accuracy:.4f}")
        
        return self.training_metadata
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições usando o modelo treinado."""
        if not self.is_trained:
            raise ValueError(f"Modelo {self.model_name} não foi treinado ainda!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades das predições (se suportado pelo modelo)."""
        if not self.is_trained:
            raise ValueError(f"Modelo {self.model_name} não foi treinado ainda!")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"Modelo {self.model_name} não suporta predict_proba")
    
    def get_current_params(self) -> Dict[str, Any]:
        """Retorna os parâmetros atuais do modelo."""
        if self.model is None:
            return self.get_default_params()
        
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Salva o modelo treinado em arquivo."""
        if not self.is_trained:
            raise ValueError("Não é possível salvar um modelo não treinado!")
        
        model_data = {
            'model': self.model,
            'metadata': self.training_metadata,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Carrega um modelo salvo."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.training_metadata = model_data['metadata']
        self.model_name = model_data['model_name']
        self.is_trained = True
        
        print(f"✅ Modelo carregado de: {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Retorna importância das features (se suportado pelo modelo)."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                name='feature_importance'
            )
        else:
            return None
    
    def __str__(self) -> str:
        status = "Treinado" if self.is_trained else "Não treinado"
        return f"{self.model_name} ({status})"
    
    def __repr__(self) -> str:
        return self.__str__()