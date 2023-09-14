import streamlit as st
import pandas as pd
import pickle

model_name = None
message = None

def init():
    with open("uploadedfiles.txt", "w") as file:
        file.write("")
    model_selection()
    st.title("Suicidal Thoughts Detection")
    st.caption('''800K people end their lives every year worldwide. Suicide is not a suddent act but it 
                  builds over time. Take the below test to see if someone close to you is at a risk of it.''')
    st.caption('''All you need is a text from the suspect. Note that we do no store any of the messages with us.''')
    

def txt_inp():
    global message
    message = st.text_input("Please provide us with the message here.")
    if message is not None and message != "":
        st.write("You said '" + message + "'")


def save_uploaded_file(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())

    with open("uploadedfiles.txt", "w") as file:
        file.write(uploadedfile.name)
    
    file.close()

def csv_inp():
    datafile = st.file_uploader("Choose a file", type= ["CSV"])
    if datafile is not None:
        save_uploaded_file(datafile)

def model_selection():
    global model_name

    model_name = st.sidebar.selectbox("Predict using", ("Naive-Bayes", "Support Vector Machine", "Recurrent Neural Network"))
    st.sidebar.write("Running on " + model_name)

def show_next_move(prediction):

    st.sidebar.write("Report generated...scroll down for results")
    if prediction == "suicide":
        st.subheader("our model predics that your behavious depicts suicidal in nature.")
        st.write("Suicide is not an option, please see a medical practitioner for counselling")
        st.write("Spending time with friends and family helps a lot")
        st.write("Or call on suicide helpline number +91-9152 98 78 21")
    else:
        st.subheader("our model predics that your behavious depicts non-suicidal in nature.")
        st.write("We are improving to detect specific behaviours...please be with us. ")
        st.write("Spending time with friends and family helps a lot...try it!")


def show_results():
    global model_name, message
    st.write("Currently Selected model: " + model_name)

    naive_model = pickle.load(open(r"E:\advance workspace\streamlit\suicidal\NaiveBayes.pkl", "rb"))
    svm_model = pickle.load(open(r"E:\advance workspace\streamlit\suicidal\SupportVector.pkl", "rb"))
    if message is not None and message != "":
        if model_name == "Naive-Bayes":
            prediction = naive_model.predict([message])[0]
            show_next_move(prediction)
        elif model_name == "Support Vector Machine":
            prediction = svm_model.predict([message])[0]
            show_next_move(prediction)



init()
txt_inp()
st.subheader("Or You can enter a csv file for bulk messages")
csv_inp()

with st.expander("See Further Details"):
    head, describe = st.tabs(["Header", "Description"])
    try:
        with open("uploadedfiles.txt", "r") as file:
            filename= file.read()
        with head:
            st.dataframe(pd.read_csv(filename).head())
        with describe:
            st.dataframe(pd.read_csv(filename).describe())

    except:
        st.write("Start with entering a csv file")

if st.sidebar.button("Show results"):
    show_results()

