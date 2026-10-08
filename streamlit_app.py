"""
Streamlit entry point forwarder.
Allows running either:
    streamlit run app.py
or:
    streamlit run streamlit_app.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app", run_name="__main__")
else:
    # If imported by streamlit
    import app
