import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import DATA_PATH, ROOT


@pytest.mark.parametrize("page, title", [
    ("overview", "Customer churn overview"), ("explorer", "Data Explorer"),
    ("prediction", "Predict Churn"), ("performance", "Model Performance"),
    ("batch", "Batch Predictions"),
])
def test_page_and_interactions(page, title):
    if page in {"overview", "explorer"} and not DATA_PATH.exists():
        pytest.skip("Run python -m scripts.download_data for local dashboard tests.")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    if page != "overview":
        app.switch_page(f"views/{page}.py")
    app.run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == title
    if page == "prediction":
        app.button[0].click().run()
        assert not app.exception
        assert not app.error
        assert app.metric[0].label == "Churn probability"
    if page == "explorer":
        app.multiselect[0].set_value([]).run()
        assert "No customers match" in app.info[0].value
