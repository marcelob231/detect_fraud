# test_decision_tree.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.decision_tree import DecisionTreeModel, create_and_train_decision_tree
import pandas as pd

def test_complete_workflow():
    """Testa o workflow completo do Decision Tree."""
    print("🌳" + "="*60)
    print("           TESTANDO DECISION TREE - WORKFLOW COMPLETO")
    print("="*63)
    
    try:
        # Método 1: Usando a função de conveniência
        print("\n🚀 MÉTODO 1: Função de conveniência")
        print("-" * 40)
        
        model = create_and_train_decision_tree()
        
        # Carregar dados de teste
        test_df = pd.read_csv("data/processed/decision_tree/test.csv")
        X_test = test_df.drop(['Class'], axis=1)
        y_test = test_df['Class']
        
        # Avaliar modelo
        evaluation = model.get_detailed_evaluation(X_test, y_test)
        
        # Análise das features mais importantes
        feature_names = X_test.columns.tolist()
        top_features = model.get_top_features(feature_names, top_n=10)
        
        print(f"\n📊 RESUMO DOS RESULTADOS:")
        print(f"   Acurácia no teste: {evaluation['accuracy']:.4f}")
        print(f"   F1-Score (Fraude): {evaluation['fraud_f1']:.4f}")
        print(f"   F1-Score (Normal): {evaluation['normal_f1']:.4f}")
        print(f"   F1-Score Macro: {evaluation['macro_avg_f1']:.4f}")
        
        return model, evaluation, top_features
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_manual_creation():
    """Testa criação manual do modelo."""
    print("\n🛠️ MÉTODO 2: Criação manual")
    print("-" * 40)
    
    try:
        # Carregar dados manualmente
        train_df = pd.read_csv("data/processed/decision_tree/train.csv")
        val_df = pd.read_csv("data/processed/decision_tree/validation.csv")
        
        X_train = train_df.drop(['Class'], axis=1)
        y_train = train_df['Class']
        X_val = val_df.drop(['Class'], axis=1)
        y_val = val_df['Class']
        
        # Criar modelo com parâmetros customizados
        model = DecisionTreeModel(
            max_depth=15,
            min_samples_split=200,
            min_samples_leaf=100,
            random_state=42
        )
        
        # Treinar
        training_info = model.train(X_train, y_train, X_val, y_val)
        
        print(f"✅ Modelo personalizado treinado!")
        print(f"   Profundidade da árvore: {training_info['tree_depth']}")
        print(f"   Número de folhas: {training_info['n_leaves']}")
        
        return model
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_predictions():
    """Testa predições em amostras específicas."""
    print("\n🎯 TESTANDO PREDIÇÕES")
    print("-" * 40)
    
    try:
        # Usar modelo já treinado
        model = create_and_train_decision_tree()
        
        # Carregar algumas amostras de teste
        test_df = pd.read_csv("data/processed/decision_tree/test.csv")
        X_test = test_df.drop(['Class'], axis=1)
        y_test = test_df['Class']
        
        # Testar em 5 amostras aleatórias
        sample_indices = [0, 100, 500, 1000, 5000]
        
        for i in sample_indices:
            if i < len(X_test):
                sample = X_test.iloc[i:i+1]
                true_label = y_test.iloc[i]
                
                prediction = model.predict(sample)[0]
                probabilities = model.predict_proba(sample)[0]
                
                print(f"   Amostra {i}:")
                print(f"     Real: {'Fraude' if true_label == 1 else 'Normal'}")
                print(f"     Predito: {'Fraude' if prediction == 1 else 'Normal'}")
                print(f"     Probabilidades: Normal={probabilities[0]:.3f}, Fraude={probabilities[1]:.3f}")
                print(f"     {'✅ Correto' if prediction == true_label else '❌ Erro'}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 INICIANDO TESTES DO DECISION TREE")
    print("="*50)
    
    # Teste 1: Workflow completo
    model, evaluation, features = test_complete_workflow()
    
    if model is not None:
        # Teste 2: Criação manual
        model2 = test_manual_creation()
        
        # Teste 3: Predições
        test_predictions()
        
        print("\n🎉 TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("="*50)
        
        if evaluation:
            print(f"🏆 RESULTADO FINAL:")
            print(f"   Acurácia: {evaluation['accuracy']:.4f}")
            print(f"   Precisão (Fraude): {evaluation['fraud_precision']:.4f}")
            print(f"   Recall (Fraude): {evaluation['fraud_recall']:.4f}")
            print(f"   F1-Score (Fraude): {evaluation['fraud_f1']:.4f}")
            
    else:
        print("❌ Testes falharam. Verifique os erros acima.")