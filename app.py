import streamlit as st

# Define pages 
home = st.Page("pages/Welcome.py", icon='🏠')

detect_trash = st.Page("pages/Detect_Trash.py", icon='🔍') # For detection

# analysis_report = st.Page("pages/Analysis_Report.py", icon='📈') # For analysis report

what_we_did = st.Page("pages/What_We_Did.py", icon='🤔') # For what we did

# Group pages
pg = st.navigation({
    "Home": [home],
    "Detect": [detect_trash], 
    "Project Walkthrough" : [what_we_did],
})

# Run the navigation
pg.run() 