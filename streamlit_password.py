import streamlit as st

# Define password
PASSWORD = "secret123"  # Change this to your desired password

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Create tabs
tab1, tab2 = st.tabs(["Public Tab", "Protected Tab"])

# Public tab (Accessible to everyone)
with tab1:
    st.header("Public Tab")
    st.write("This tab is accessible to everyone.")

# Protected tab (Requires authentication)
with tab2:
    if not st.session_state.authenticated:
        st.header("Protected Tab 🔒")
        password = st.text_input("Enter Password:", type="password")
        
        if st.button("Login"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")
    else:
        st.header("Welcome to the Protected Tab! ✅")
        st.write("This tab contains sensitive information.")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
