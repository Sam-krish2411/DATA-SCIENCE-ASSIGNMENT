import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ===============================
# Load and Prepare Titanic Dataset
# ===============================
titanic = pd.read_csv(r"C:\Users\DELL\Downloads\DS_LOGISTICS\Titanic_train.csv")

# Drop missing values
titanic = titanic.dropna(subset=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Encode categorical variables
le_sex = LabelEncoder()
titanic['Sex'] = le_sex.fit_transform(titanic['Sex'])   # male=1, female=0

le_embarked = LabelEncoder()
titanic['Embarked'] = le_embarked.fit_transform(titanic['Embarked'])

# Features (X) and Target (Y)
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = titanic['Survived']

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X, Y)

# ===============================
# Streamlit App
# ===============================
st.title("🚢 Titanic Survival Prediction App")

st.sidebar.header("Enter Passenger Details")

def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', (1, 2, 3))
    Sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    Age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    SibSp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.sidebar.number_input("Ticket Fare", min_value=0.0, value=32.0)
    Embarked = st.sidebar.selectbox('Port of Embarkation', ('C', 'Q', 'S'))

    # Convert categorical to numeric same as training
    Sex_num = 1 if Sex == 'male' else 0
    Embarked_num = le_embarked.transform([Embarked])[0]

    data = {
        'Pclass': Pclass,
        'Sex': Sex_num,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked_num
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display Results
st.subheader('Predicted Result')
st.write('Survived' if prediction[0] == 1 else 'Did Not Survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)