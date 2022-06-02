import streamlit as st
from modules.decisionmaking import *
import pandas as pd
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
accounts = {
    'admin': 'admin',
    'test': 'test',
}


def is_logged_in():
    return 'valid_user' in st.session_state and st.session_state['valid_user']


def is_valid_account(username, password):
    return username in accounts.keys() and password == accounts[username]


def display_login():
    st.title("Login")
    form = st.form('form')
    username = form.text_input(label='Username', key='username')
    password = form.text_input(label='Password', type='password', key='password')
    if form.form_submit_button('Login'):
        if is_valid_account(username, password):
            st.session_state['valid_user'] = True
            st.experimental_rerun()
        else:
            form.error('Invalid credentials!')



def run():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Final Year Project By Sathish, Swetha S, Swetha P'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    if not is_logged_in():
        display_login()
    else:
        Main.goto()


run()
    








