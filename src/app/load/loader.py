import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import json
import logging

class DataLoader:
    """
    Carregador principal de dados para todos os modelos de ML.
    Responsável por carregar, dividir e preparar dados de forma estratificada.
    """
    
    def __init__(self, data_path: str = "data/raw/creditcard_2023.csv"):
        self.data_path = Path(data_path)
        self.processed_path = Path("data/processed")
        self.metadata_path = Path("data/processed/metadata")
        
        # Garantir que diretórios existem
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.metadata = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        """Carrega dados brutos do CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
            
        print(f"Carregando dados de {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Criar metadados básicos
        self.metadata = {
            "total_records": len(self.data),
            "total_columns": len(self.data.columns),
            "fraud_count": int(self.data['Class'].sum()),
            "normal_count": int(len(self.data) - self.data['Class'].sum()),
            "fraud_percentage": float(self.data['Class'].mean() * 100),
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict()
        }
        
        print(f"✅ Dados carregados: {len(self.data):,} registros")
        print(f"   - Normal: {self.metadata['normal_count']:,} ({100-self.metadata['fraud_percentage']:.2f}%)")
        print(f"   - Fraude: {self.metadata['fraud_count']:,} ({self.metadata['fraud_percentage']:.2f}%)")
        
        return self.data
    
    def create_stratified_splits(
        self, 
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Cria divisões estratificadas dos dados.
        
        Args:
            train_size: Proporção para treinamento (padrão: 60%)
            val_size: Proporção para validação (padrão: 20%)  
            test_size: Proporção para teste (padrão: 20%)
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com (train_df, val_df, test_df)
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("A soma das proporções deve ser 1.0")
            
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_raw_data() primeiro.")
        
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Primeira divisão: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(val_size + test_size),
            stratify=y,
            random_state=random_state
        )
        
        # Segunda divisão: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=random_state
        )
        
        # Recriar DataFrames completos
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Atualizar metadados
        self.metadata['splits'] = {
            'train': {
                'size': len(train_df),
                'fraud_count': int(y_train.sum()),
                'fraud_percentage': float(y_train.mean() * 100)
            },
            'validation': {
                'size': len(val_df),
                'fraud_count': int(y_val.sum()),
                'fraud_percentage': float(y_val.mean() * 100)
            },
            'test': {
                'size': len(test_df),
                'fraud_count': int(y_test.sum()),
                'fraud_percentage': float(y_test.mean() * 100)
            }
        }
        
        print("✅ Divisões estratificadas criadas:")
        for split_name, info in self.metadata['splits'].items():
            print(f"   {split_name}: {info['size']:,} registros ({info['fraud_percentage']:.2f}% fraude)")
        
        return train_df, val_df, test_df
    
    def save_splits_for_model(
        self, 
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Dict[str, Path]:
        """
        Salva as divisões de dados para um modelo específico.
        
        Args:
            model_name: Nome do modelo (ex: 'decision_tree')
            train_df, val_df, test_df: DataFrames das divisões
            
        Returns:
            Dicionário com os caminhos dos arquivos salvos
        """
        model_dir = self.processed_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        files = {
            'train': model_dir / 'train.csv',
            'validation': model_dir / 'validation.csv',
            'test': model_dir / 'test.csv'
        }
        
        # Salvar CSVs
        train_df.to_csv(files['train'], index=False)
        val_df.to_csv(files['validation'], index=False)
        test_df.to_csv(files['test'], index=False)
        
        # Salvar metadados específicos do modelo
        model_metadata = {
            **self.metadata,
            'model_name': model_name,
            'files': {name: str(path) for name, path in files.items()},
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.metadata_path / f'{model_name}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ Dados salvos para {model_name}:")
        for name, path in files.items():
            print(f"   {name}: {path}")
        print(f"   metadata: {metadata_file}")
        
        return files
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calcula class weights para lidar com desbalanceamento.
        
        Returns:
            Dicionário com pesos das classes {0: peso_normal, 1: peso_fraude}
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_raw_data() primeiro.")
        
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.data['Class'])
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=self.data['Class']
        )
        
        class_weights = dict(zip(classes, weights))
        
        print("✅ Class weights calculados:")
        print(f"   Normal (0): {class_weights[0]:.3f}")
        print(f"   Fraude (1): {class_weights[1]:.3f}")
        
        return class_weights


# Função de conveniência para uso rápido
def prepare_data_for_decision_tree(data_path: str = "data/raw/creditcard_2023.csv"):
    """
    Função de conveniência para preparar dados especificamente para Decision Tree.
    
    Returns:
        Tuple com (arquivos_salvos, class_weights, metadados)
    """
    loader = DataLoader(data_path)
    
    # Carregar dados
    loader.load_raw_data()
    
    # Criar divisões estratificadas
    train_df, val_df, test_df = loader.create_stratified_splits()
    
    # Salvar arquivos para decision tree
    files = loader.save_splits_for_model('decision_tree', train_df, val_df, test_df)
    
    # Calcular class weights
    class_weights = loader.get_class_weights()
    
    return files, class_weights, loader.metadata


if __name__ == "__main__":
    # Teste rápido
    files, weights, metadata = prepare_data_for_decision_tree()
    print("\n🎯 Preparação concluída para Decision Tree!")