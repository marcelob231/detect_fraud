# test_loader.py
import sys
import os

# Adicionar o diretório src ao Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from app.load.loader import prepare_data_for_decision_tree

if __name__ == "__main__":
    print("🚀 Testando DataLoader...")
    print("=" * 50)
    
    try:
        # Preparar dados para Decision Tree
        files, class_weights, metadata = prepare_data_for_decision_tree()
        
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print(f"📊 Total de registros: {metadata['total_records']:,}")
        print(f"🎯 Proporção de fraude: {metadata['fraud_percentage']:.3f}%")
        
        print(f"\n⚖️ Class Weights:")
        print(f"   Normal: {class_weights[0]:.3f}")
        print(f"   Fraude: {class_weights[1]:.3f}")
        
        print(f"\n📁 Arquivos criados:")
        for name, path in files.items():
            print(f"   {name}: {path}")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()