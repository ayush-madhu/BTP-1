import streamlit as st
from static_mode import static_mode
from dynamic_mode import dynamic_mode

# Program starts here
if __name__ == "__main__":

    # Set the title of the app
    st.markdown("<h1 style='text-align: center;'>Drying Analyzer</h1>", unsafe_allow_html=True)

    # Create a hamburger menu using selectbox
    option = st.selectbox(
        "Select Mode",
        ["Select Mode", "Static", "Dynamic"],
        index=0,
    )
    
    if option == "Static":
        static_mode()
    
    if option == "Dynamic":
        dynamic_mode()