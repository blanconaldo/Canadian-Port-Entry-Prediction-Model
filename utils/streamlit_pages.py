import streamlit as st

def init_session_state():
    """Initialize session state variables."""
    if 'pages_visited' not in st.session_state:
        st.session_state.pages_visited = set()
    if 'df' not in st.session_state:
        st.session_state.df = None

def check_page_prerequisites(page_number):
    """Check if all prerequisites are met for a given page."""
    init_session_state()

    # Define prerequisites for each page
    prerequisites = {
        0: [],    # Home page
        1: [0],   # Data Metrics needs Home
        2: [0,1], # Data Preprocessing needs Home and Data Metrics
        3: [0,1,2],
        4: [0,1,2,3],
        5: [0,1,2,3,4],
        6: [0,1,2,3,4,5],
        7: [0,1,2,3,4,5,6],
        8: [0,1,2,3,4,5,6,7],
        9: [0,1,2,3,4,5,6,7,8]
    }

    if page_number not in prerequisites:
        return False

    return all(prereq in st.session_state.pages_visited for prereq in prerequisites[page_number])

def add_page_number(page_number):
    """Add page number to visited pages."""
    init_session_state()
    st.session_state.pages_visited.add(page_number)
