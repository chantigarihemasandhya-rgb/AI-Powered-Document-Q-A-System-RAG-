def test_project_files_exist():
    from pathlib import Path
    assert Path("app/rag.py").exists()
    assert Path("app/api.py").exists()
    assert Path("streamlit_app.py").exists()
