# explore_decision_tree.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.decision_tree import DecisionTreeModel, create_and_train_decision_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

def create_output_dir():
    """Cria diretório para salvar os resultados."""
    output_dir = Path("results/decision_tree_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def comprehensive_analysis():
    """Análise completa do Decision Tree com visualizações."""
    print("🔍" + "="*60)
    print("     ANÁLISE COMPLETA DO DECISION TREE")
    print("="*63)
    
    output_dir = create_output_dir()
    
    # Treinar modelo
    print("\n🚀 Treinando modelo...")
    model = create_and_train_decision_tree()
    
    # Carregar dados de teste
    test_df = pd.read_csv("data/processed/decision_tree/test.csv")
    X_test = test_df.drop(['Class'], axis=1)
    y_test = test_df['Class']
    feature_names = X_test.columns.tolist()
    
    # Avaliação detalhada
    print("\n📊 Avaliação detalhada...")
    evaluation = model.get_detailed_evaluation(X_test, y_test)
    
    # 1. GRÁFICO DE IMPORTÂNCIA DAS FEATURES
    print("\n🎯 Criando gráfico de importância das features...")
    fig1 = model.plot_feature_importance(
        feature_names, 
        top_n=15, 
        figsize=(12, 8),
        save_path=str(output_dir / "feature_importance.png")
    )
    plt.close()
    
    # 2. ESTRUTURA DA ÁRVORE (primeiros níveis)
    print("\n🌳 Criando visualização da estrutura da árvore...")
    fig2 = model.plot_tree_structure(
        feature_names,
        max_depth=3,
        figsize=(20, 12),
        save_path=str(output_dir / "tree_structure.png")
    )
    plt.close()
    
    # 3. MATRIZ DE CONFUSÃO DETALHADA
    print("\n📈 Criando matriz de confusão...")
    create_confusion_matrix_plot(evaluation, output_dir)
    
    # 4. DISTRIBUIÇÃO DAS PROBABILIDADES
    print("\n📊 Analisando distribuição das probabilidades...")
    analyze_prediction_probabilities(model, X_test, y_test, output_dir)
    
    # 5. ANÁLISE DAS TOP FEATURES
    print("\n🏆 Analisando distribuição das top features...")
    analyze_top_features(model, X_test, y_test, feature_names, output_dir)
    
    # 6. ANÁLISE DE CASOS ESPECÍFICOS
    print("\n🔍 Analisando casos específicos...")
    analyze_specific_cases(model, X_test, y_test, feature_names, output_dir)
    
    # 7. SALVAR MODELO
    print("\n💾 Salvando modelo...")
    model_path = output_dir / "decision_tree_model.joblib"
    model.save_model(str(model_path))
    
    # 8. RELATÓRIO FINAL
    create_final_report(model, evaluation, output_dir)
    
    print(f"\n✅ ANÁLISE COMPLETA CONCLUÍDA!")
    print(f"📁 Resultados salvos em: {output_dir}")
    
    return model, evaluation

def create_confusion_matrix_plot(evaluation, output_dir):
    """Cria gráfico da matriz de confusão."""
    cm = np.array(evaluation['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'])
    plt.title('Matriz de Confusão - Decision Tree')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    
    # Adicionar percentuais
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_prediction_probabilities(model, X_test, y_test, output_dir):
    """Analisa a distribuição das probabilidades de predição."""
    probabilities = model.predict_proba(X_test)
    fraud_probs = probabilities[:, 1]  # Probabilidades de fraude
    
    # Separar por classe real
    normal_probs = fraud_probs[y_test == 0]
    fraud_probs_real = fraud_probs[y_test == 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma das probabilidades
    ax1.hist(normal_probs, bins=50, alpha=0.7, label='Transações Normais', color='blue')
    ax1.hist(fraud_probs_real, bins=50, alpha=0.7, label='Transações Fraudulentas', color='red')
    ax1.set_xlabel('Probabilidade de Fraude')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição das Probabilidades de Fraude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [normal_probs, fraud_probs_real]
    ax2.boxplot(data_to_plot, labels=['Normal', 'Fraude'])
    ax2.set_ylabel('Probabilidade de Fraude')
    ax2.set_title('Box Plot - Probabilidades por Classe')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "probability_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Estatísticas
    print(f"   📊 Probabilidades - Transações Normais:")
    print(f"      Média: {normal_probs.mean():.4f}")
    print(f"      Mediana: {np.median(normal_probs):.4f}")
    print(f"      Máximo: {normal_probs.max():.4f}")
    
    print(f"   📊 Probabilidades - Transações Fraudulentas:")
    print(f"      Média: {fraud_probs_real.mean():.4f}")
    print(f"      Mediana: {np.median(fraud_probs_real):.4f}")
    print(f"      Mínimo: {fraud_probs_real.min():.4f}")

def analyze_top_features(model, X_test, y_test, feature_names, output_dir):
    """Analisa a distribuição das features mais importantes."""
    # Obter top 6 features
    importance = model.get_feature_importance()
    importance.index = feature_names
    top_6_features = importance.nlargest(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_6_features):
        ax = axes[i]
        
        # Dados por classe
        normal_data = X_test[y_test == 0][feature]
        fraud_data = X_test[y_test == 1][feature]
        
        # Histogramas
        ax.hist(normal_data, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        ax.hist(fraud_data, bins=50, alpha=0.7, label='Fraude', color='red', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Densidade')
        ax.set_title(f'{feature} (Importância: {importance[feature]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_features_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_specific_cases(model, X_test, y_test, feature_names, output_dir):
    """Analisa casos específicos - acertos e erros."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Encontrar casos interessantes
    correct_predictions = predictions == y_test
    incorrect_predictions = ~correct_predictions
    
    # Fraudes detectadas com baixa confiança
    fraud_detected_low_conf = (predictions == 1) & (y_test == 1) & (probabilities[:, 1] < 0.8)
    
    # Falsos positivos com alta confiança
    false_positives_high_conf = (predictions == 1) & (y_test == 0) & (probabilities[:, 1] > 0.8)
    
    print(f"   📊 Estatísticas de Casos:")
    print(f"      Total de testes: {len(y_test):,}")
    print(f"      Predições corretas: {correct_predictions.sum():,} ({correct_predictions.mean()*100:.2f}%)")
    print(f"      Predições incorretas: {incorrect_predictions.sum():,} ({incorrect_predictions.mean()*100:.2f}%)")
    print(f"      Fraudes detectadas c/ baixa confiança: {fraud_detected_low_conf.sum()}")
    print(f"      Falsos positivos c/ alta confiança: {false_positives_high_conf.sum()}")
    
    # Analisar alguns casos específicos se existirem
    if fraud_detected_low_conf.sum() > 0:
        print(f"\n   🔍 Exemplo de fraude detectada com baixa confiança:")
        idx = np.where(fraud_detected_low_conf)[0][0]
        sample = X_test.iloc[idx:idx+1]
        print(f"      Probabilidade de fraude: {probabilities[idx, 1]:.3f}")
        
        # Analisar caminho de decisão
        decision_path = model.get_decision_path(sample)
        print(f"      Caminho de decisão: {len(decision_path['decision_path'])} nós")

def create_final_report(model, evaluation, output_dir):
    """Cria relatório final em texto."""
    report_path = output_dir / "analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("        RELATÓRIO DE ANÁLISE - DECISION TREE\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. MÉTRICAS DE PERFORMANCE:\n")
        f.write("-"*30 + "\n")
        f.write(f"Acurácia: {evaluation['accuracy']:.4f}\n")
        f.write(f"Precisão (Fraude): {evaluation['fraud_precision']:.4f}\n")
        f.write(f"Recall (Fraude): {evaluation['fraud_recall']:.4f}\n")
        f.write(f"F1-Score (Fraude): {evaluation['fraud_f1']:.4f}\n")
        f.write(f"F1-Score Macro: {evaluation['macro_avg_f1']:.4f}\n\n")
        
        f.write("2. CARACTERÍSTICAS DO MODELO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Profundidade da árvore: {model.training_metadata['tree_depth']}\n")
        f.write(f"Número de folhas: {model.training_metadata['n_leaves']}\n")
        f.write(f"Features utilizadas: {model.training_metadata['n_features_used']}\n")
        f.write(f"Tempo de treinamento: {model.training_metadata['training_time_seconds']:.2f}s\n\n")
        
        f.write("3. MATRIZ DE CONFUSÃO:\n")
        f.write("-"*30 + "\n")
        cm = evaluation['confusion_matrix']
        f.write(f"Verdadeiros Negativos: {cm[0][0]:,}\n")
        f.write(f"Falsos Positivos: {cm[0][1]:,}\n")
        f.write(f"Falsos Negativos: {cm[1][0]:,}\n")
        f.write(f"Verdadeiros Positivos: {cm[1][1]:,}\n\n")
        
        f.write("4. ARQUIVOS GERADOS:\n")
        f.write("-"*30 + "\n")
        f.write("• feature_importance.png - Importância das features\n")
        f.write("• tree_structure.png - Estrutura da árvore\n")
        f.write("• confusion_matrix.png - Matriz de confusão\n")
        f.write("• probability_distribution.png - Distribuição das probabilidades\n")
        f.write("• top_features_distribution.png - Distribuição das top features\n")
        f.write("• decision_tree_model.joblib - Modelo salvo\n")
    
    print(f"   📄 Relatório salvo em: {report_path}")

if __name__ == "__main__":
    model, evaluation = comprehensive_analysis()
    
    print("\n🎉 ANÁLISE COMPLETA FINALIZADA!")
    print("="*40)
    print("Verifique a pasta 'results/decision_tree_analysis/' para:")
    print("• Gráficos de visualização")
    print("• Modelo treinado salvo")
    print("• Relatório detalhado")