import streamlit as st
from backend import StockForecastModel
from login import AuthSystem

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
    choice = st.sidebar.radio("Select Option", tabs, key="auth_tabs_radio")


    if choice == "Login":
        auth.login()
    else:
        auth.signup()


