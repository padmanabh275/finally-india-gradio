"""
FinAlly India – Hugging Face Space app

Slimmed-down Streamlit entrypoint suitable for deploying as a Space.
Uses the same core UI and logic as streamlit_app.py but relies on
HF-managed environment variables and hardware.
"""

from streamlit_app import main


if __name__ == "__main__":
    main()

