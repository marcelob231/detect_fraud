# test_api.py
import requests
import json

# Configurações
BASE_URL = 'http://localhost:5000'
headers = {'Content-Type': 'application/json'}

def test_health_check():
    """Testa o health check da API."""
    print("🔍 Testando Health Check...")
    response = requests.get(f'{BASE_URL}/api/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_auth_info():
    """Testa informações de autenticação."""
    print("\n🔍 Testando Auth Info...")
    response = requests.get(f'{BASE_URL}/api/auth/info')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_login():
    """Testa login e retorna token."""
    print("\n🔑 Testando Login...")
    login_data = {
        'username': 'admin',
        'password': 'admin123'
    }
    
    response = requests.post(
        f'{BASE_URL}/api/auth/login',
        headers=headers,
        json=login_data
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    if response.status_code == 200:
        token = result.get('access_token')
        print(f"✅ Token obtido: {token[:50]}...")
        return token
    
    return None

def test_protected_routes(token):
    """Testa rotas protegidas com token."""
    if not token:
        print("❌ Não é possível testar rotas protegidas sem token")
        return
    
    auth_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    print("\n🔒 Testando rotas protegidas...")
    
    # Testar profile
    print("- Profile:")
    response = requests.get(f'{BASE_URL}/api/auth/profile', headers=auth_headers)
    print(f"  Status: {response.status_code}")
    
    # Testar lista de modelos
    print("- Lista de modelos:")
    response = requests.get(f'{BASE_URL}/api/models/', headers=auth_headers)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print(f"  Modelos: {response.json()}")
    
    # Testar info do Decision Tree
    print("- Info Decision Tree:")
    response = requests.get(f'{BASE_URL}/api/models/decision_tree/info', headers=auth_headers)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print(f"  Info: {response.json()}")

def main():
    """Executa todos os testes."""
    print("🧪 INICIANDO TESTES DA API")
    print("=" * 50)
    
    # Testes básicos (sem autenticação)
    health_ok = test_health_check()
    auth_info_ok = test_auth_info()
    
    if not health_ok:
        print("❌ API não está funcionando. Verifique se o servidor está rodando.")
        return
    
    # Testes de autenticação
    token = test_login()
    
    if token:
        test_protected_routes(token)
        print("\n✅ TODOS OS TESTES BÁSICOS CONCLUÍDOS!")
    else:
        print("\n❌ Falha na autenticação")
    
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Execute o servidor: python src/app/main.py")
    print("2. Teste as APIs no browser ou Postman")
    print("3. Use as credenciais: admin/admin123")

if __name__ == "__main__":
    main()