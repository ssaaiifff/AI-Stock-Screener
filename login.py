import streamlit as st
import pandas as pd
import os

class AuthSystem:
    def __init__(self):
        self.file_path = "users.csv"
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=["username", "password"])
            df.to_csv(self.file_path, index=False)
        
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.username = ""
    
    def load_users(self):
        return pd.read_csv(self.file_path)
    
    def save_users(self, df):
        df.to_csv(self.file_path, index=False)
    
    def login(self):
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users_df = self.load_users()
            if ((users_df["username"] == username) & (users_df["password"] == password)).any():
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    def signup(self):
        st.subheader("Create a New Account")
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Sign Up"):
            users_df = self.load_users()
            if new_username in users_df["username"].values:
                st.error("Username already exists. Choose a different one.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                new_user = pd.DataFrame({"username": [new_username], "password": [new_password]})
                users_df = pd.concat([users_df, new_user], ignore_index=True)
                self.save_users(users_df)
                st.success("Account created successfully! You can now log in.")

st.title("Welcome to AI Stock Screener")
auth = AuthSystem()
tabs = ["Login", "Signup"]
choice = st.sidebar.radio("Select Option", tabs)

if st.session_state.authenticated:
    st.success(f"You are logged in as {st.session_state.username}")
    st.write("**Stock Screener Dashboard Coming Soon...**")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()
else:
    if choice == "Login":
        auth.login()
    else:
        auth.signup()

