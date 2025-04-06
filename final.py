import streamlit as st
from backend import StockForecastModel
from login import AuthSystem

def set_background():
    page_bg_img = '''
    <style>
    .stApp {
        background: linear-gradient(to right, #3366ff, #ff99cc);
        color: black;
    }

    /* Glass-style buttons */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.3);
        color: black;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid white;
        color: #000000;
    }

    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .logout-container {
        padding-top: 12px;
        padding-right: 5px;
    }

    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply background
set_background()

# Initialize authentication
auth = AuthSystem()

# Show app if user is authenticated
if st.session_state.authenticated:
    # Top title and logout button in one row
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("### 📊 Stock Price Forecasting")
    with col2:
        with st.container():
            st.markdown('<div class="logout-container">', unsafe_allow_html=True)
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Run stock app
    app = StockForecastModel()
    app.run()

else:
    auth.login_signup_screen()
