from backend.app.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.weaviate_class_name == "PropertyDocChunk"
    assert settings.ollama_chat_model
