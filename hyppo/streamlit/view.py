import streamlit as st


# Add a selectbox to the sidebar:
select_event = st.sidebar.selectbox(
    'Please select the artefact to work with',
    ('Hypotheses and Models', 'Domain Ontology', 'Workflow', 'Research Lattice', 'Configuration', 'Virtual Experiment')
)

if select_event == 'Hypotheses and Models':
    st.title('Hypotheses and Models')
    st.write('Hypotheses')

elif select_event == 'Domain Ontology':
    st.title('Domain Ontology')

elif select_event == 'Workflow':
    st.title('Workflow')

elif select_event == 'Research Lattice':
    st.title('Research Lattice')

elif select_event == 'Configuration':
    st.title('Configuration')

else:
    st.title('Virtual Experiment')


