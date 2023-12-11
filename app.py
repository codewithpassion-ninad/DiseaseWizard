import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time



train = pd.read_csv('./Dataset/Training.csv')
test = pd.read_csv('./Dataset/Training.csv')

train = train.drop(["Unnamed: 133"], axis=1)

label_encoder = LabelEncoder()
train['prognosis'] = label_encoder.fit_transform(train['prognosis'])
test['prognosis'] = label_encoder.transform(test['prognosis'])

X_train = train.drop(['prognosis'], axis=1)
y_train = train['prognosis']
X_test = test.drop(['prognosis'], axis=1)
y_test = test['prognosis']

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

page_by_img="""
<style>
[data-testid="stAppViewContainer"]{
background:linear-gradient(90deg, rgba(55,49,173,1) 14%, rgba(0,0,0,1) 65%, rgba(72,72,72,1) 100%);
}
</style>
"""
page_side="""
<style>
[data-testid="stSidebar"]{
background: radial-gradient(circle, rgba(8,18,69,1) 0%, rgba(80,1,17,1) 100%);
</style>
"""
st.markdown(page_side,unsafe_allow_html=True)
st.markdown(page_by_img,unsafe_allow_html=True)
for model_name, model in models.items():
    model.fit(X_train, y_train)

st.markdown("<h1 style='text-align:center; color: lightgray'>DISEASE WIZARD🩺👨‍💻</h1><br><br>", unsafe_allow_html=True)
# sidebar_width = 300
# st.markdown(
#     f"""
#     <style>
#         .reportview-container .main .block-container {{
#             max-width: {sidebar_width}px;
#             padding-top: 2rem;
#             padding-right: 1rem;
#             padding-left: 1rem;
#             padding-bottom: 2rem;
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
with st.sidebar:
    st.markdown("""
    <style>
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 300px;
            padding: 20px;
            background-color: #f4f4f4;
            overflow-y: auto;

        }
    </style>
""", unsafe_allow_html=True)
    st.markdown("## **Unlocking the Power of AI in Healthcare** 🌐🤖")

    st.write("Welcome to Disease Wizard, where cutting-edge artificial intelligence meets healthcare exploration. Our journey to predict diseases involves a fusion of advanced Machine Learning (ML) models and your unique symptoms.")

    st.markdown("### **1. AI at Your Service**")
    st.write("Disease Wizard is fueled by highly efficient ML models, meticulously trained to understand patterns and correlations within a vast dataset of health information. These models are the wizards behind the curtain, ready to assist you on your health discovery.")

    st.markdown("### **2. Your Symptoms, Your Key**")
    st.write("Select your symptoms from the list provided – your input is the magic spell that activates the AI. The more accurate and detailed your symptoms, the more precise the predictions!")

    st.markdown("### **3. The Dance with Data**")
    st.write("Behind the scenes, your selected symptoms dance with our extensive dataset, where the AI models have learned the intricate choreography of health patterns. It's a harmonious collaboration between your unique health fingerprint and the wisdom distilled from vast amounts of medical data.")

    st.markdown("### **4. Prediction Unveiled**")
    st.write("As you hit the 'Predict' button, the AI springs into action, predicting potential diseases based on the synergy between your symptoms and the trained models.")

    st.error("Go and try it out 🚀✨")

symptoms = list(X_train.columns)

# Create a multiselect box for user input
selected_symptoms = st.multiselect('Select your symptoms', symptoms)

# Check if at least three symptoms are selected
if len(selected_symptoms) < 3:
    st.warning("Please select at least three symptoms before predicting.")
else:
    # Create a dictionary to store the user input
    user_input = {}
    with st.spinner("Loading..."):
    # Simulate some time-consuming computation
        time.sleep(2)
    # Loop through the symptoms and assign values based on user selection
    for symptom in symptoms:
        # Assign 1 if the symptom is selected, 0 otherwise
        user_input[symptom] = 1 if symptom in selected_symptoms else 0

    # Convert the user input to a data frame
    input_df = pd.DataFrame([user_input])


    # Create a button to make predictions
    if st.button('Predict'):
        unique_predictions = []
        # Loop through the models and make predictions
        for i, (model_name, model) in enumerate(models.items()):
            # Predict the probability of each class
            proba = model.predict_proba(input_df)[0]
            # Get the index of the highest probability
            pred = np.argmax(proba)
            # Get the name of the predicted class
            pred_name = label_encoder.inverse_transform([pred])[0]
            if pred_name not in unique_predictions:
                # Add the predicted class to the unique predictions list
                unique_predictions.append(pred_name)
                # Display the prediction result
                st.markdown(f"<h4 style='color:magenta;'>Prediction {len(unique_predictions)}</h4>",unsafe_allow_html=True)
                st.write(f'Chance of having {pred_name}')
        st.success('Predictions have been generated successfully!', icon="✅")
footer = """
<style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 100%;
        background: radial-gradient(circle, rgba(8, 18, 69, 1) 0%, rgba(80, 1, 17, 1) 100%);
        padding:1%;
        color: white;
        text-align: center;
    }
   
</style>
<div class="footer">
    <p>🩺 Disease Wizard 🧙‍♂ | Developed by Ninad Sugandhi and Vasu Bhasin 👨‍💻👨‍💻</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)