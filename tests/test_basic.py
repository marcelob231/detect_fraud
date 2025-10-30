from src.app.main import create_app

def test_index_route():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"message": "Hello from your Python model!"}