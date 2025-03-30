import streamlit as st
from backend import StockForecastModel
from login import AuthSystem

def set_background():
    page_bg_img = '''
        <style>
        .stApp {
            background: linear-gradient(to right, #3366ff, #ff99cc);
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: black !important;
        }
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply background
set_background()

# Title
st.title("Welcome to AI Stock Screener")

# Initialize authentication
auth = AuthSystem()

# Show app if user is authenticated
if st.session_state.authenticated:
    app = StockForecastModel()
    app.run()

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()
else:
    auth.login_signup_screen()  # ✅ This is the ONLY place where login widget is created
