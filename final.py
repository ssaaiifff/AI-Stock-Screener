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
set_background()
st.title("Welcome to AI Stock Screener")

# Initialize authentication system
auth = AuthSystem()

# If the user is logged in, show the stock forecasting app
if st.session_state.authenticated:
    app = StockForecastModel()
    app.run()

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()
else:
    tabs = ["Login", "Signup"]
    choice = st.sidebar.radio("Select Option", tabs)

    if choice == "Login":
        auth.login()
    else:
        auth.signup()


