import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title = 'Credit Default App'
    #page_icon = "C"
)

@st.cache()
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data.dropna())

@st.cache()
def load_model():
    filename = "finalized_default_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

data = load_data()
model = load_model()

st.title("Sharky's Credit Default App")

st.write('This Application is a a dashboard to visualize credit defaults and making predictions')

row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])

rate = row1_col1.slider("Interst the customer has to pay",
data["borrower_rate"].min(),
data["borrower_rate"].max(),
(0.07, 0.15)
)

row1_col1.write(rate)

income = row1_col2.slider("Monthly income",
data["monthly_income"].min(),
data["monthly_income"].max(),
(20000.0, 30000.0)
)

row1_col2.write(income)

mask= ~data.columns.isin(["loan_default", "employment_status", "borrower_rate"])
names = data.iloc[:, mask].columns

variable = row1_col3.selectbox("Select Variables to Compare", names)
row1_col3.write(variable)

filter_data = data.loc[(data["borrower_rate"]>=rate[0]) &
(data["borrower_rate"]<=rate[1]) &
(data["monthly_income"]>= income[0]) &
(data["monthly_income"]<= income[1])
]

if st.checkbox("Show filtered data", False):
    st.subheader("raw data")
    st.write(filter_data)


row2_col1, row2_col2 = st.columns([1,1])

barplotdata = filter_data[["loan_default", variable]].groupby("loan_default").mean()

fig1, ax = plt.subplots(figsize=(8, 3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color="#fc8d62")
ax.set_ylabel(variable)

row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1)

fig2 = sns.lmplot(y="borrower_rate", x=variable, data=filter_data)

row2_col2.subheader("Borrower Rate Correlations")
row2_col2.pyplot(fig2)

st.header("Predicting Customer Default")

uploaded_data = st.file_uploader("Choose a file with customer data for masking predictiongs")

st.write(uploaded_data)
if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    
    predictions = model.predict(new_customers)
    new_customers["predictions"] = predictions

    st.download_button(label="Download Scored Customer Data",
    data=new_customers.to_csv().encode("utf-8"),
    file_name="scored_new_customers")
