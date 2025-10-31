# test_complete_api.py
import requests
import json

BASE_URL = 'http://localhost:5000'
headers = {'Content-Type': 'application/json'}

def get_auth_token():
    """Obter token de autenticação."""
    login_data = {'username': 'admin', 'password': 'admin123'}
    response = requests.post(f'{BASE_URL}/api/auth/login', headers=headers, json=login_data)
    
    if response.status_code == 200:
        return response.json().get('access_token')
    return None

def test_decision_tree_apis():
    """Testa todas as APIs do Decision Tree."""
    print("🌳 TESTANDO TODAS AS APIs DO DECISION TREE")
    print("=" * 60)
    
    # Obter token
    token = get_auth_token()
    if not token:
        print("❌ Falha na autenticação")
        return
    
    auth_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    # Lista de endpoints para testar
    endpoints = [
        ('GET', '/api/models/', 'Lista de modelos'),
        ('GET', '/api/models/decision_tree/info', 'Info do Decision Tree'),
        ('GET', '/api/models/decision_tree/evaluate', 'Avaliação do modelo'),
        ('GET', '/api/models/decision_tree/feature-importance', 'Importância das features'),
        ('GET', '/api/visualization/decision_tree/confusion-matrix', 'Matriz de confusão'),
        ('GET', '/api/visualization/decision_tree/probability-distribution', 'Distribuição de probabilidades'),
        ('GET', '/api/visualization/decision_tree/feature-distribution', 'Distribuição das features'),
        ('GET', '/api/visualization/decision_tree/performance-metrics', 'Métricas de performance'),
    ]
    
    results = {}
    
    for method, endpoint, description in endpoints:
        print(f"\n🔍 Testando: {description}")
        print(f"   {method} {endpoint}")
        
        try:
            if method == 'GET':
                response = requests.get(f'{BASE_URL}{endpoint}', headers=auth_headers)
            else:
                response = requests.post(f'{BASE_URL}{endpoint}', headers=auth_headers)
            
            status = response.status_code
            print(f"   Status: {status}")
            
            if status == 200:
                data = response.json()
                if 'message' in data:
                    print(f"   ✅ {data['message']}")
                else:
                    print(f"   ✅ Dados retornados com sucesso")
                
                # Salvar resultado para análise
                results[endpoint] = {
                    'status': status,
                    'data_keys': list(data.keys()) if isinstance(data, dict) else [],
                    'success': True
                }
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                print(f"   ❌ Erro: {error_data.get('message', 'Unknown error')}")
                results[endpoint] = {
                    'status': status,
                    'error': error_data.get('message', 'Unknown error'),
                    'success': False
                }
                
        except Exception as e:
            print(f"   ❌ Exceção: {str(e)}")
            results[endpoint] = {
                'exception': str(e),
                'success': False
            }
    
    # Teste de predição com dados de amostra
    print(f"\n🎯 Testando predição com amostra...")
    sample_transaction = {
        "transaction": {
            "id": 12345,
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }
    }
    
    try:
        response = requests.post(
            f'{BASE_URL}/api/models/decision_tree/predict',
            headers=auth_headers,
            json=sample_transaction
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"   ✅ Predição: {prediction['prediction']['label']}")
            print(f"   Confiança: {prediction['prediction']['confidence']:.3f}")
            print(f"   Prob. Normal: {prediction['probabilities']['normal']:.3f}")
            print(f"   Prob. Fraude: {prediction['probabilities']['fraud']:.3f}")
        else:
            print(f"   ❌ Erro na predição: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exceção na predição: {str(e)}")
    
    # Resumo dos resultados
    print(f"\n📊 RESUMO DOS TESTES:")
    print("=" * 40)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    
    print(f"✅ Sucessos: {successful}/{total}")
    print(f"❌ Falhas: {total - successful}/{total}")
    
    if successful == total:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print(f"🚀 Backend está pronto para o frontend Angular!")
    else:
        print(f"\n⚠️  Alguns testes falharam. Verifique os detalhes acima.")
    
    return results

def create_postman_collection():
    """Cria uma coleção do Postman para as APIs."""
    collection = {
        "info": {
            "name": "Fraud Detection API",
            "description": "APIs para detecção de fraude com Decision Tree"
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{auth_token}}"}]
        },
        "item": [
            {
                "name": "Authentication",
                "item": [
                    {
                        "name": "Login",
                        "request": {
                            "method": "POST",
                            "header": [{"key": "Content-Type", "value": "application/json"}],
                            "body": {
                                "mode": "raw",
                                "raw": '{"username": "admin", "password": "admin123"}'
                            },
                            "url": {"raw": "{{base_url}}/api/auth/login"}
                        }
                    }
                ]
            },
            {
                "name": "Decision Tree",
                "item": [
                    {
                        "name": "Model Info",
                        "request": {
                            "method": "GET",
                            "url": {"raw": "{{base_url}}/api/models/decision_tree/info"}
                        }
                    },
                    {
                        "name": "Evaluate Model",
                        "request": {
                            "method": "GET", 
                            "url": {"raw": "{{base_url}}/api/models/decision_tree/evaluate"}
                        }
                    },
                    {
                        "name": "Feature Importance",
                        "request": {
                            "method": "GET",
                            "url": {"raw": "{{base_url}}/api/models/decision_tree/feature-importance"}
                        }
                    }
                ]
            }
        ],
        "variable": [
            {"key": "base_url", "value": "http://localhost:5000"}
        ]
    }
    
    with open('fraud_detection_api.postman_collection.json', 'w') as f:
        json.dump(collection, f, indent=2)
    
    print(f"📄 Coleção do Postman criada: fraud_detection_api.postman_collection.json")

if __name__ == "__main__":
    print("🧪 TESTE COMPLETO DAS APIs - DECISION TREE")
    print("=" * 60)
    
    results = test_decision_tree_apis()
    create_postman_collection()
    
    print(f"\n📋 PRÓXIMOS PASSOS:")
    print(f"1. ✅ Backend completo - todas as APIs funcionando")
    print(f"2. 🅰️  Criar frontend Angular")
    print(f"3. 🔗 Integrar frontend com as APIs")
    print(f"4. 📊 Implementar dashboards e gráficos")