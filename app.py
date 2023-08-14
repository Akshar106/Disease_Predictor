import streamlit as st
import pandas as pd
import os
from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt
#import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from img_classification import brain_tumor_detection
from keras.models import Sequential
from keras.layers import Dense, Flatten
import cv2 as cv
import tensorflow as tf
import re



import joblib
import cv2
#👋
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Home',
                           'Diabetes Prediction',
                           'Heart Diseases Prediction',
                           'Brain Tumor Detection',
                           'Pneumonia Detection'],
                          icons=['house','activity','heart'],
                          default_index=0)

#--------------------------------------------------------------->>Home<<---------------------------------------------------------------------
selected_option = None
if selected == 'Home':

    def main():

        st.markdown("<h1 style='color: #FF5733';><em>Welcome to Diseases Predictor!🏥</em></h1>", unsafe_allow_html=True)
        st.image("Medical Prediction Home Page.png")
        st.markdown("<h2 style='color: #FF5733';><em>Overview :</em></h2>", unsafe_allow_html=True)
        st.write('A disease predictor web app is a platform that utilizes various algorithms and machine learning techniques to analyze input data and predict the likelihood of a user having a specific disease or medical condition. These web app is designed to provide users with insights into their health status based on the information they provide, such as symptoms, medical history, lifestyle habits, and demographic data.')
        st.markdown("<h2 style='color: #FF5733';><em>Information about the Diseases which this webApp can predict :</em></h2>", unsafe_allow_html=True)
        options = ['Diabetes Diseases', 'Brain Tumor', 'Heart Diseases','Pneumonia Detection']
        selected_option = st.radio('', options)
    
        if selected_option == 'Diabetes Diseases':
            st.markdown("<h2 style='color: #FF5733';><em>Diabetes :</em></h2>", unsafe_allow_html=True)
            st.write('Diabetes is a chronic medical condition characterized by high levels of glucose (sugar) in the blood. It occurs when the body either does not produce enough insulin (a hormone that regulates blood sugar) or is unable to effectively use the insulin it produces. As a result, glucose accumulates in the bloodstream, leading to various health complications.')
            st.markdown("<h2 style='color: #FF5733';><em>There are two primary types of Diabetes :</em></h2>", unsafe_allow_html=True)
            st.image('diabtypes.png')
            #st.write('1. **Type 1 Diabetes:** This is an autoimmune condition where the bodys immune system attacks and destroys the insulin-producing cells in the pancreas. As a result, people with Type 1 diabetes require insulin injections or an insulin pump to manage their blood sugar levels.')
            #st.write('2. **Type 2 Diabetes:** This is the most common form of diabetes and is typically associated with lifestyle factors such as obesity, physical inactivity, and unhealthy eating habits. In Type 2 diabetes, the body becomes resistant to insulin, and the pancreas may not produce enough insulin to compensate. It can often be managed through lifestyle changes, medication, and, in some cases, insulin therapy.')
            st.markdown("<h2 style='color: #FF5733';><em>Symptoms :</em></h2>", unsafe_allow_html=True)
            #st.title('Unordered List Example')
            # List items for the unordered list
            st.image('diabhomesymptoms.jpg')
            st.markdown("<h2 style='color: #FF5733';><em>How to manage Diabetes :</em></h2>", unsafe_allow_html=True)
            list_items = ['Managing diabetes involves a combination of lifestyle modifications, regular blood sugar monitoring, medications, and in some cases, insulin therapy.', 'Proper management helps prevent complications and allows individuals with diabetes to lead healthy and active lives.', 'It is essential for people with diabetes to work closely with healthcare professionals to develop a personalized diabetes management plan that includes a balanced diet, regular exercise, and self-monitoring of blood sugar levels.', 'Education and support from healthcare providers can empower individuals with diabetes to make informed decisions about their health and well-being.']
            # Create an unordered list using HTML tags
            unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
            # Display the unordered list using st.markdown()
            st.markdown(unordered_list, unsafe_allow_html=True)
            #st.write('**Ans** - Managing diabetes involves a combination of lifestyle modifications, regular blood sugar monitoring, medications, and in some cases, insulin therapy. Proper management helps prevent complications and allows individuals with diabetes to lead healthy and active lives. It is essential for people with diabetes to work closely with healthcare professionals to develop a personalized diabetes management plan that includes a balanced diet, regular exercise, and self-monitoring of blood sugar levels. Education and support from healthcare providers can empower individuals with diabetes to make informed decisions about their health and well-being.')
            st.markdown('For More Information : [Click Here](https://diabetes.org/)')



        elif selected_option == 'Brain Tumor':
            st.markdown("<h2 style='color: #FF5733';><em>Brain Tumor :</em></h2>", unsafe_allow_html=True)
            st.write('A brain tumor is an abnormal growth or mass of cells within the brain. Brain tumors can develop in different parts of the brain and can be either benign (non-cancerous) or malignant (cancerous). They can originate from brain tissue itself (primary brain tumors) or spread to the brain from other parts of the body (metastatic brain tumors).')
            st.markdown("<h2 style='color: #FF5733';><em>Types of Brain Tumor :</em></h2>", unsafe_allow_html=True)
            st.image('braintumortype.jpg')
            st.write('1. **Gliomas** : Gliomas are the most common types of brain tumors and priginate from glial cells , which provide support and nourishment to neurons. There are different subtypes of gliomas , including astrocytomas  ,oligodendrogliomas , and ependymomas. Gliomas can rnage form low-grade (slow growing) to higher-grade (aggresive and fast growing). ')
            st.write('2. **Meningiomas** : Meningiomas develop in the menings which are the protectivee membrances covering the brain and spinal cord. These tumors are usualy slow-growing and often benign . However, depending on their size and location ,they can sometimes cause symptomps due to pressure on the brain.')
            st.write('3. **Pituitary Adenomas** : These Tumors arise form the pituitary gland , a small gland located at the base of the brain. Most pituitary adeomas are benign, but they can cause hormone imbalances and neurological symptoms depending on the homones they secrete and their size.')
            st.write('4. **Medulloblastomas** : Medulloblastomas are type of embryonal tumor that typically occurs in the cerebellum, which is responsible for coordinating movement. They are more common in children and are consider malignant.')
            st.write('5. **Schwannomas**: Schwannomas arise from Schwann cells, which form the protective covering (myelin sheath) around nerves. They are usually benign and commonly affect the nerves associated with balance and hearing (e.g., vestibular schwannoma or acoustic neuroma).')
            st.write('6. **Craniopharyngiomas**: Craniopharyngiomas are rare, usually benign tumors that form near the pituitary gland. They can cause various hormonal imbalances and may affect vision and other functions.')
            st.markdown("<h2 style='color: #FF5733';><em>Symptoms :</em></h2>", unsafe_allow_html=True)
            st.image('braintumorsymptoms.jpg')
            #list_items = ['Headaches.', 'Seizures.', 'Nausea and vomiting.', 'Cognitive and personality changes.', 'Vision problems.' , 'Speech difficulties.' , 'Weakness or numbness.' , 'Changes in sensation' , 'Fatigue.']
            # Create an unordered list using HTML tags
            #unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
            # Display the unordered list using st.markdown()
            #st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown("<h2 style='color: #FF5733';><em>How to manage Brain Tumor :</em></h2>", unsafe_allow_html=True)
            list_items = ['Managing a brain tumor typically involves a multidisciplinary approach.', 'Accurate diagnosis is essential for determining the type, size, location, and grade of the tumor. This is usually done through imaging techniques like MRI or CT scans, along with a biopsy, if necessary.', 'Seek help from a team of medical professionals experienced in brain tumor treatment. This may include neurologists, neuro-oncologists, neurosurgeons, radiation oncologists and other specialists.', 'Surgery : In some cases, surgery may be performed to remove as much of the tumor as possible without causing damage to critical brain regions.', 'Radation Therapy : herapyRadation therapy uses hight-energy X-rays to target and destroy tumor cells. It is often used after surgery to trat any remaining tumor cells or as the primary treatment for tumors that are difficult to remove surgically.' , 'Chemotherapy : Chemotherapy involves using drugs to kill or slow the growth of cancer cells. It can be administered orally or intravenously and may be used in conbination with other treatments.' , 'Immunotherapy : Immuno Therapy aims to boost the bodys immune system to recognize and attack cancer cells. It is an area of ongoing research and may be used in some cases.' , 'Clinical Trails : Clinical trials offer oppurtunities to access new treatments and therapies that are still being researched. Patients and their medical tema may consider participating in appropriate clinical trials.' , 'Regular follow up appointments are crucial to monitor the tumors progress and the patients overall health. Adjustments to the treatment plan may be based on the tumors response to the therapy.']

            unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
            # Display the unordered list using st.markdown()
            st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown('For More Information : [Click Here](https://www.cancer.gov/types/brain)')
  
  
  
        elif selected_option == 'Heart Diseases':
            st.markdown("<h2 style='color: #FF5733';><em>Heart Diseases :</em></h2>", unsafe_allow_html=True)
            st.write('Heart diseases, also known as cardiovascular diseases (CVD), refer to a group of medical conditions that affect the heart and blood vessels. They are a significant global health concern and remain one of the leading causes of death and disability worldwide. Heart diseases encompass a wide range of conditions, each with its unique characteristics and implications.')
            st.markdown("<h2 style='color: #FF5733';><em>Some common types of Heart Diseases Include :</em></h2>", unsafe_allow_html=True)
            st.image('hearttypes.jpg')
            st.write('1. **Coronary Artery Disease (CAD):** CAD is the most prevalent form of heart disease. It occurs when plaque buildup narrows the coronary arteries, reducing blood flow to the heart muscle. This can lead to chest pain (angina) or result in a heart attack if a blood clot forms and completely blocks the artery.')
            st.write('2. **Heart Attack (Myocardial Infarction):** A heart attack happens when a coronary artery becomes severely blocked, preventing blood flow to a section of the heart muscle. This causes the affected heart muscle to die if not promptly treated.')
            st.write('3. **Heart Failure:** Heart failure occurs when the hearts ability to pump blood efficiently is compromised. This can happen gradually over time due to conditions like CAD, hypertension, or other heart-related problems.')
            st.write('4. **Hypertensive Heart Disease:** Prolonged high blood pressure can lead to damage and strain on the heart, resulting in hypertensive heart disease.')
            st.write('5. **Congenital Heart Defects:** Some individuals are born with structural abnormalities in the heart, known as congenital heart defects. These defects can affect the hearts function and blood circulation.')

            st.markdown("<h2 style='color: #FF5733';><em>Symptoms :</em></h2>", unsafe_allow_html=True)
            st.image('heartsymptoms.jpg')
            #list_items = ['Chest Pain or Discomfort.', 'Shortness of Breath.', 'Fatigue.', 'Dizziness and Fainting.' , 'Swelling.' ' Cold Sweats.']
                # Create an unordered list using HTML tags
            #unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
            #st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown("<h2 style='color: #FF5733';><em>How to control Heart Diseases :</em></h2>", unsafe_allow_html=True)

            list_items = ['Preventive measures for heart diseases include adopting a healthy lifestyle, including regular exercise, a balanced diet, avoiding smoking and excessive alcohol consumption, and managing stress. ', 'Early detection, proper medical management, and following a healthcare providers recommendations are crucial in reducing the risk of complications associated with heart diseases', 'Regular check-ups and screenings can help identify risk factors and underlying conditions, enabling timely intervention and better management of heart health.']
                # Create an unordered list using HTML tags
            unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
            st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown('For More Information : [Click Here](https://www.cdc.gov/heartdisease/index.htm)')



        elif selected_option == 'Pneumonia Detection' :
            st.markdown("<h2 style='color: #FF5733';><em>Pneumonia :</em></h2>", unsafe_allow_html=True)
            st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs, causing them to fill with fluid or pus. The infection can be caused by bacteria, viruses, fungi, or other microorganisms. Pneumonia can range from mild to severe, and it can be life-threatening, especially for young children, older adults, and people with weakened immune systems.')
            st.markdown("<h2 style='color: #FF5733';><em>Some common types of Pneumonia Include :</em></h2>", unsafe_allow_html=True)
            st.image('pneutypes.jpg')
            st.write('1. **Community-Acquired Pneumonia (CAP)** : This is the most common type of pneumonia and occurs outside of healthcare settings. It is caused by various microorganisms, including bacteria, viruses, and less commonly, fungi.')
            st.write('2. **Bacterial Pneumonia** : Bacterial pneumonia is caused by bacterial infections, and Streptococcus pneumoniae (pneumococcus) is the most common bacteria responsible for this type of pneumonia. Other bacteria, such as Haemophilus influenzae and Staphylococcus aureus, can also cause bacterial pneumonia.')
            st.write('3. **Viral Pneumonia** :  Viral pneumonia is caused by different viruses, with influenza viruses (flu) and respiratory syncytial virus (RSV) being common culprits. It is more prevalent in children and older adults.')
            st.write('4. **Mycoplasma Pneumonia** : Mycoplasma pneumonia, also known as atypical or walking pneumonia, is caused by Mycoplasma pneumoniae, a type of bacteria. It is usually milder but can persist for an extended period.')
            st.write('5. **Fungal Pneumonia** : Fungal pneumonia is caused by various fungi, and it primarily affects individuals with weakened immune systems, such as those with HIV/AIDS or undergoing cancer treatment.')

            st.markdown("<h2 style='color: #FF5733';><em>Symptoms :</em></h2>", unsafe_allow_html=True)
            st.image('pneusymptoms.jpg')
            #list_items = ['Cough.', 'Fever.', 'Shortness of breath.', 'Chest pain.' , 'Fatigue.' 'Sweating and shaking.']
                # Create an unordered list using HTML tags
            #unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
            #st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown("<h2 style='color: #FF5733';><em>How to control Pneumonia :</em></h2>", unsafe_allow_html=True)
            list_items = ['Vaccines are an essential part of pneumonia prevention. Getting vaccinated can protect against specific pathogens that cause pneumonia, especially in high-risk populations.', ' Proper handwashing with soap and water, especially after coughing or sneezing, can reduce the spread of viruses and bacteria that cause respiratory infections, including pneumonia.', 'Smoking damages the lungs and weakens the bodys defense against respiratory infections. Avoiding smoking and exposure to secondhand smoke can lower the risk of pneumonia.', 'Eating a balanced diet, staying physically active, and getting enough sleep can help strengthen the immune system, making the body better equipped to fight off infections.' , 'o prevent aspiration pneumonia, especially in vulnerable individuals such as the elderly or those with swallowing difficulties, it is essential to take precautions while eating and drinking. These precautions may include sitting upright while eating, eating slowly, and avoiding lying down immediately after eating.' 'Infants, young children, older adults, pregnant women, and individuals with weakened immune systems are at higher risk for pneumonia. Extra care and preventive measures should be taken for these groups.','For individuals with chronic health conditions such as diabetes, heart disease, or lung disease, managing these conditions effectively can reduce the risk of pneumonia.','In indoor environments, ensuring good ventilation and air quality can help reduce the risk of respiratory infections.']
                # Create an unordered list using HTML tags
            unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
            st.markdown(unordered_list, unsafe_allow_html=True)

            st.markdown('For More Information : [Click Here](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204)')

        
    if __name__ == "__main__":
        main()


#----------------------------------------------->> Diabetes Prediction <<------------------------------------------------


if selected == 'Diabetes Prediction':
    df = pd.read_csv('diabetes-dataset.csv')

# HEADINGS
    st.markdown("<h2 style='color: #FF5733';>Diabetes Checkup:</h2>", unsafe_allow_html=True)

# Add activity icon after the title
    #icon_url = 'https://icons.getbootstrap.com/icons/activity/'
    #st.image(icon_url, caption='Activity Icon',use_column_width=True)
    #st.markdown("<i class='fas fa-running'></i> activity", unsafe_allow_html=True)

    st.sidebar.header('Patient Data')
    #st.subheader('Training Data Stats')
    #st.write(df.describe())

# X AND Y DATA
    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
    def predict_diabetes_type(user_data):

    # Sample conditions for demonstration purposes

        glucose_level = user_data['Glucose'].values[0]
        insulin_level = user_data['Insulin'].values[0]
        blood_pressure = user_data['BloodPressure'].values[0]
        bmi = user_data['BMI'].values[0]
        pregnancies = user_data['Pregnancies'].values[0]
        age = user_data['Age'].values[0]

        if glucose_level < 100:
            if bmi < 25 and age >= 40:
                return "Type 2 Diabetes"
            else:
                return "Type 1 Diabetes"
        elif glucose_level >= 100 and glucose_level < 126 and insulin_level < 200:
            if age >= 40:
                return "Type 2 Diabetes"
            else:
                return "Type 3 Diabetes"
        else:
            if insulin_level > 200:
                return "Type 1 Diabetes"
            else:
                return "Type 2 Diabetes"
    def get_widget_key(widget_name):
    # Generate a unique key for each widget based on its name
        return hash(widget_name)

    def user_report():
        Pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 17, 3, key=get_widget_key('Pregnancies'))
        Glucose = st.sidebar.slider('Enter Your Glucose Level in (mg/dL)', 0, 400, 120, key=get_widget_key('Glucose'))
        BloodPressure = st.sidebar.slider('Enter Your Blood Pressure in (mm Hg)', 0, 122, 70, key=get_widget_key('BloodPressure'))
        SkinThickness = st.sidebar.slider('Enter Your Skin Thickness Value in (mm)', 0, 100, 20, key=get_widget_key('SkinThickness'))
        Insulin = st.sidebar.slider('Enter Your Insulin Level in (mu U/ml)', 0, 846, 79, key=get_widget_key('Insulin'))
        BMI = st.sidebar.slider('Enter Your Body Mass Index (BMI) in kg/(height in m)^2)', 0.0, 67.0, 20.0, step=0.1, key=get_widget_key('BMI'))
        DiabetesPedigreeFunction = st.sidebar.slider('Enter Your Diabetes Pedigree Function Value', 0.0, 2.4, 0.47, key=get_widget_key('DPF'))
        Age = st.sidebar.slider('Enter Your Age', 21, 88, 33, key=get_widget_key('Age'))


        user_report_data = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        
         }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data
    
    

# PATIENT DATA
    user_data = user_report()
    st.markdown("<h3 style='color: #FF7F50;'><em>Patient Data : </em></h3>", unsafe_allow_html=True)
    st.write(user_data)
   
# MODEL
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    user_data_processed = user_data[x_train.columns]
    user_result = rf.predict(user_data)

    diab_diagnosis=""
    diabetes_type = ""
    
    if st.button('Diabetes Test Result'):
        if user_result[0] == 0:
            diab_diagnosis = 'You are not Diabetic'
        elif user_result[0] == 1:
            diab_diagnosis = 'You are Diabetic. Immediately Consult a Doctor'
            #diabetes_type = predict_diabetes_type(user_data)

    st.success(diab_diagnosis)
    if st.button('Generalized Report'):
        #report_data = user_report()
        pregnancies_value = user_data['Pregnancies'].iloc[0]
        glucose_value = user_data['Glucose'].iloc[0]
        bp_value = user_data['BloodPressure'].iloc[0]
        insulin_level = user_data['Insulin'].iloc[0]
        bmi_value = user_data['BMI'].iloc[0]

       
        gulcose_range = '70 - 99'
        bp_range = '90/60 - 120/80'
        insuline_range = '16 - 166'
        bmi_range = '18.5 - 24.9'
        if glucose_value < 70:
            glucose_flag = 'Low'
        elif glucose_value > 99:
            glucose_flag = 'High'
        else:
            glucose_flag = 'Normal'

        if bp_value < 60:
            bp_flag = 'Low'
        elif bp_value > 120:
            bp_flag = 'High'
        else:
            bp_flag = 'Normal'

        if insulin_level > 166:
            insulin_flag = 'High'
        elif insulin_level < 16 :
            insulin_flag = 'Low'
        else :
            insulin_flag = 'Normal'

        if bmi_value < 18.5:
            bmi_flag = 'Low'
        elif bmi_value > 24.9:
            bmi_flag = 'High'
        else:
            bmi_flag = 'Normal'
        

        data = {    
        'Features' : ['Glucose Level', 'Blood Pressure', 'Insulin Level', 'Body Mass Index'],
        'Report Values' : [ glucose_value, bp_value, insulin_level, bmi_value],
        'Status' : [ glucose_flag , bp_flag , insulin_flag , bmi_flag],
        'Normal_Range' : [ gulcose_range, bp_range, insuline_range, bmi_range],
        'Unit' : [ 'mg/dL' , 'mm Hg' , 'μU/mL' ,'kg/m^2']
        }
        

# Combine the features, values, and normal range into a single line
        

# Display the formatted table using st.markdown
        #st.subheader('Viz Starts Here')
        def process_range(range_str):
            match = re.findall(r'\d+\.\d+|\d+', range_str)
            values = [float(val) for val in match]
            return sum(values) / len(values)

# Function to plot a specific feature
        def plot_feature(feature_name, report_value, normal_range, unit):
            plt.figure(figsize=(4, 3))
            plt.bar(['Report Value', 'Normal Range'], [report_value, normal_range], color=['#FF7F50', 'g'])
            plt.ylabel(f'{feature_name} ({unit})')
            #plt.title(f'{feature_name} Report Value vs Normal Range')

    # Show the chart within Streamlit
            st.pyplot(plt)
        df = pd.DataFrame(data)

# Convert DataFrame to tabular format with proper alignment and spacing
        def format_table(dataframe):
            html_table = "<table style='width:100%; text-align:center; border: 1px solid white; border-collapse: collapse;'>"
    # Header row
            html_table += "<tr style='border: 1px solid white;'><th style='border: 1px solid white;color: #FF7F50;font-size: 30px;'>Features</th><th style='border: 1px solid white;color: #FF7F50;font-size: 30px;'>Report Values</th><th style='border: 1px solid white;color: #FF7F50;font-size: 30px;'>Status<th style='border: 1px solid white;color: #FF7F50;font-size: 30px;'>Normal Range<th style='border: 1px solid white;color: #FF7F50;font-size: 30px;'>Units</th></tr>"
    # Data rows
            for index, row in dataframe.iterrows():
                html_table += "<tr style='border: 1px solid white;'>"
                for col in dataframe.columns:
                    html_table += f"<td style='border: 1px solid white;'>{row[col]}</td>"
                html_table += "</tr>"
            html_table += "</table>"
            return html_table
        st.markdown(format_table(df), unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

  
        for idx, feature_name in enumerate(data['Features']):
                report_value = data['Report Values'][idx]
                normal_range = process_range(data['Normal_Range'][idx])
                unit = data['Unit'][idx]

                st.markdown(f'<h3 style="color: #FF5733;"><em>{feature_name} Visualization</em></h3>', unsafe_allow_html=True)
                plot_feature(feature_name, report_value, normal_range, unit)

        
       

    
    # Show the "Click to know the type of Diabetes" button only when the user is predicted to have diabetes
    if user_result[0] == 1:
        
        if st.button('Click to know the type of Diabetes'):
            diabetes_type = predict_diabetes_type(user_data)
            st.write('Diabetes Type : ',diabetes_type)
            if(diabetes_type == 'Type 1 Diabetes'):
                st.markdown("<h2 style='color: #FF5733;'><em>What is Type 1 Diabetes?</em></h2>", unsafe_allow_html=True)
                list_items = ['It is also known as "Insulin-Dependent Diabetes" or "Juvenile Diabetes".', 'Type 1 diabetes is an autoimmune disease where the bodys immune system attacks and destroys the insulin-producing beta cells in the pancreas.', 'This results in little to no insulin production, leading to high blood glucose levels.', 'It often develops in childhood or early adulthood, but it can occur at any age.' , 'People with Type 1 diabetes require lifelong insulin therapy for survival.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h2 style='color: #FF5733;'><em>Symptoms of Type 1 Diabetes</em></h2>", unsafe_allow_html=True)
                st.image('diabsymptoms.png')


                st.markdown("<h2 style='color: #FF5733;'><em>How to Control Type 1 Diabetes?</em></h2>", unsafe_allow_html=True)
                st.write('Type 1 diabetes is a chronic condition where the body does not produce insulin. Since insulin is essential for regulating blood sugar levels, individuals with Type 1 diabetes need to take insulin through injections or an insulin pump to survive. ')
                st.write('While it cannot be cured, Type 1 diabetes can be managed effectively to lead a healthy and fulfilling life. Here are some key strategies for controlling Type 1 diabetes :')
                st.image('diabcontrol.jpg')
                
                st.markdown("<h6 style='color:   #FF7F50;'><em>1 - Insulin Therapy : </em></h6>", unsafe_allow_html=True)
                list_items = ['Insulin is the mainstay of treatment for Type 1 diabetes.', 'Type 1 diabetes is an autoimmune disease where the bodys immune system attacks and destroys the insulin-producing beta cells in the pancreas.', 'It is essential to take the prescribed insulin as directed by the healthcare provider.', 'There are various types of insulin with different onset and duration of action.' , 'Insulin doses may be adjusted based on blood glucose levels, carbohydrate intake, physical activity, and other factors.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)


                st.markdown("<h6 style='color:  #FF7F50;'><em>2 - Blood Glucose Monitoring : </em></h6>", unsafe_allow_html=True)
                list_items = ['Regular monitoring of blood glucose levels is crucial for managing Type 1 diabetes.', 'This helps in understanding how different factors, such as food, physical activity, and stress, affect blood sugar levels.', 'Continuous Glucose Monitoring (CGM) devices can provide real-time glucose readings, which offer better control and prevent severe high or low blood sugar episodes.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>4 - Carbohydreate Counting : </em></h6>", unsafe_allow_html=True)
                list_items = ['Carbohydrates have the most significant impact on blood sugar levels.','Learning to count carbohydrates in meals and matching insulin doses accordingly helps in maintaining stable blood glucose levels after eating.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>4 - Healthy Eating and Regular Physical Acitivity : </em></h6>", unsafe_allow_html=True)
                list_items = ['Following a balanced and nutritious diet is essential for managing Type 1 diabetes. ','Focus on whole foods, fruits, vegetables, lean proteins, and whole grains.','Avoid sugary and processed foods, as they can cause rapid spikes in blood sugar levels.','Regular exercise helps improve insulin sensitivity and can help stabilize blood sugar levels.','It is essential to find a physical activity that one enjoys and to discuss an exercise plan with a healthcare provider.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>5 - Regular Medical Checkups : </em></h6>", unsafe_allow_html=True)
                list_items = ['Regular visits to a healthcare provider are essential for monitoring blood sugar levels, assessing overall health, adjusting insulin doses, and addressing any concerns or complications.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>6 - Continuous Care : </em></h6>", unsafe_allow_html=True)
                list_items = [' Diabetes management is an ongoing process.','Regular follow-ups with healthcare providers help to monitor progress, make necessary adjustments, and address any challenges that may arise.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)


                #st.markdown("<h2 style='color: lightgreen;'>This is a light green subheader</h2>", unsafe_allow_html=True)
                #st.markdown("<p style='color: yellow;'>This is a green text</p>", unsafe_allow_html=True)



                st.markdown("<h2 style='color: #FF5733;'><em>Which Medicine one should take to Control Type 1 Diabetes?</em></h2>", unsafe_allow_html=True)
                st.write('Type 1 diabetes is a condition where the body does not produce insulin, so insulin therapy is the primary and essential treatment for individuals with Type 1 diabetes.')
                st.write('There is no medication that can replace the need for insulin in people with Type 1 diabetes. Insulin is crucial for regulating blood glucose levels and allowing glucose to enter cells for energy.')

                st.write('There are several types of insulin available, broadly classified into the following categories :')
                st.markdown("<h6 style='color:  #FF7F50;'><em>1 - Rapid-acting insulin</em></h6>", unsafe_allow_html=True)
                st.write("This type of insulin starts working quickly, usually within 15 minutes after injection, and its effects last for about 2 to 4 hours. It is often used before meals to cover the rise in blood sugar after eating.")

                st.markdown("<h6 style='color: #FF7F50;'><em>2 - Short-acting insulin</em></h6>" , unsafe_allow_html=True)
                st.write('Short-acting insulin starts working within 30 minutes to an hour after injection and remains effective for about 5 to 8 hours. It is typically used before meals.')

                st.markdown("<h6 style='color: #FF7F50;'><em>3 - Intermediate-acting insulin</em></h6>" , unsafe_allow_html=True)
                st.write('This type of insulin takes longer to start working (about 1 to 2 hours) but has a more extended duration of action (about 12 to 18 hours). It is often used to provide background or basal insulin coverage between meals and overnight.')

                st.markdown("<h6 style='color: #FF7F50;'><em>4 - Long-acting insulin</em></h6>" , unsafe_allow_html=True)
                st.write('Long-acting insulin starts working within 1 to 2 hours after injection and can provide background insulin coverage for up to 24 hours. It is used to maintain consistent blood sugar levels throughout the day.')

                st.markdown("<h6 style='color: #FF7F50;'><em>5 - Premixed insulin</em></h6>" , unsafe_allow_html=True)
                st.write('Some insulin preparations combine both rapid-acting and intermediate-acting insulin in one injection. They are typically taken before meals to cover immediate blood sugar needs and provide some background coverage.')

                st.markdown("<h2 style='color: #FF5733;'><em>How to control Type 1 Diabetes without Medicines?</em></h2>", unsafe_allow_html=True)
                st.image('diabcontrolwithoutmedicine.jpg')
                #list_items = ['Eliminate Refined Carbohydrates from Diet.','Feed on to low Glcaemic Foods.','Drink Enough Water.','Avoid Stress.','Get Adequate Sleep.','Avoid Smoking']
                # Create an unordered list using HTML tags
                #unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                #t.markdown(unordered_list, unsafe_allow_html=True)
                
                st.markdown("<h3 style='color:  #FFC175;'><em>Note : </em></h2>", unsafe_allow_html=True)
                st.write('It is essential for individuals with Type 1 diabetes to work closely with their healthcare providers to determine the most appropriate type of insulin, dosing regimen, and timing based on their individual needs, lifestyle, and blood glucose monitoring results. Insulin doses may need to be adjusted based on factors such as carbohydrate intake, physical activity, stress levels, and illness.')
                
                st.markdown('For More Information of Type 1 Diabetes : [Click Here](https://www.mayoclinic.org/diseases-conditions/type-1-diabetes/symptoms-causes/syc-20353011)')





            elif diabetes_type == 'Type 2 Diabetes' :
                st.markdown("<h2 style='color: #FF5733;'><em>What is Type 2 Diabetes?</em></h2>", unsafe_allow_html=True)
                list_items = ['It is also known as "Non-Insulin-Dependent Diabetes".', 'Type 2 diabetes is the most common form of diabetes, accounting for approximately 90-95% of all diabetes cases.', 'It is characterized by insulin resistance, where the bodys cells do not respond properly to insulin, and impaired insulin secretion by the pancreas.', 'Type 2 diabetes is often associated with lifestyle factors such as obesity, physical inactivity, and unhealthy eating habits.' , 'It is more commonly diagnosed in adulthood, but the prevalence is increasing among younger populations due to rising obesity rates.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)
                st.markdown("<h2 style='color: #FF5733;'><em>Symptoms of Type 2 Diabetes</em></h2>", unsafe_allow_html=True)
                st.image('diabtype2.jpg')

                st.markdown("<h2 style='color: #FF5733;'><em>How to control Type 2 Diabetes?</em></h2>", unsafe_allow_html=True)
                st.write('Management of type 2 diabetes typically includes a combination of lifestyle changes, medication, and monitoring. Treatment goals aim to keep blood sugar levels within a target range to reduce the risk of complications.')
                st.image('diabtype2control.png')
                st.markdown("<h6 style='color:  #FF7F50;'><em>1 - Healthy Eating : </em></h6>", unsafe_allow_html=True)
                list_items = [' Follow a well-balanced diet that focuses on whole foods, including vegetables, fruits, lean proteins, whole grains, and healthy fats.','Avoid sugary beverages, processed foods, and excessive consumption of refined carbohydrates.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>2 - Portion Control : </em></h6>", unsafe_allow_html=True)
                list_items = [' Be mindful of portion sizes to avoid overeating and help manage blood sugar levels.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>3 - Regular Physical Activity : </em></h6>", unsafe_allow_html=True)
                list_items = [' Engage in regular exercise, such as walking, swimming, cycling, or any other physical activity you enjoy.','Aim for at least 150 minutes of moderate-intensity aerobic activity per week, spread across several days.','Exercise helps improve insulin sensitivity and can aid in weight management.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>4 - Weight Management : </em></h6>", unsafe_allow_html=True)
                list_items = ['  If you are overweight or obese, losing weight can significantly improve your diabetes control. ','Even a modest weight loss of 5-10% of your body weight can make a difference.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>5 - Monitor Blood Sugar Levels : </em></h6>", unsafe_allow_html=True)
                list_items = ['  Regularly check your blood glucose levels as advised by your healthcare provider.  ',' This will help you understand how different foods, activities, and medications affect your blood sugar.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>6 - Take Medications as Prescribed : </em></h6>", unsafe_allow_html=True)
                list_items = [' If your doctor has prescribed medications, take them as instructed.  ',' This may include oral medications or insulin injections, depending on your individual needs.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>7 - Stress Management : </em></h6>", unsafe_allow_html=True)
                list_items = ['  Chronic stress can impact blood sugar levels. ','Practice relaxation techniques, such as deep breathing, meditation, yoga, or hobbies you enjoy.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>8 - Quit Smoking and Limit Alcohol Intake : </em></h6>", unsafe_allow_html=True)
                list_items = ['If you smoke, quitting can improve your overall health and diabetes management.',' If you drink alcohol, do so in moderation. ','Alcohol can cause blood sugar levels to fluctuate.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                st.markdown("<h6 style='color:  #FF7F50;'><em>9 - Regular Check-ups : </em></h6>", unsafe_allow_html=True)
                list_items = ['Schedule regular check-ups with your healthcare team, including your doctor, diabetes educator, and dietitian. ',' They can provide support and make adjustments to your treatment plan as needed.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

                
                st.markdown("<h2 style='color: #FF5733;'><em>Which Medicine one should take to Control Type 2 Diabetes?</em></h2>", unsafe_allow_html=True)
                st.write('The choice of medication for controlling type 2 diabetes depends on various factors, including the individuals overall health, blood sugar levels, response to lifestyle changes, and the presence of other medical conditions. The decision is made by a healthcare provider, typically a doctor or endocrinologist, after a thorough evaluation of the patients condition.')
                st.write(' Here are some common types of medications used to treat type 2 diabetes :')

                st.markdown("<h6 style='color:  #FF7F50;'><em>1 - Metformin : </em></h6>", unsafe_allow_html=True)
                st.write('Metformin is often the first-line medication for type 2 diabetes. It works by reducing the livers glucose production and improving insulin sensitivity in the body. Metformin is usually well-tolerated and has a long safety record.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>2 - Sulfonylureas : </em></h6>", unsafe_allow_html=True)
                st.write('hese medications stimulate the pancreas to produce more insulin. Examples include gliclazide, glipizide, and glyburide. They can be effective but may cause hypoglycemia (low blood sugar) if not taken correctly.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>3 - Meglitinides : </em></h6>", unsafe_allow_html=True)
                st.write(' Similar to sulfonylureas, meglitinides stimulate insulin release from the pancreas but have a shorter duration of action. Repaglinide and nateglinide are examples of meglitinides.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>4 - Dipeptidyl Peptidase-4 (DPP-4) Inhibitors : </em></h6>", unsafe_allow_html=True)
                st.write('DPP-4 inhibitors help increase insulin production and reduce glucose production by blocking the action of the DPP-4 enzyme. Sitagliptin, saxagliptin, and linagliptin are common DPP-4 inhibitors.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>5 - Thiazolidinediones (TZDs) : </em></h6>", unsafe_allow_html=True)
                st.write('TZDs improve insulin sensitivity in muscle and fat tissues. Pioglitazone and rosiglitazone are examples of TZDs. However, TZDs have some side effects and are usually not used as first-line therapy.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>6 - SGLT-2 Inhibitors : </em></h6>", unsafe_allow_html=True)
                st.write('Sodium-glucose co-transporter 2 (SGLT-2) inhibitors work by reducing glucose reabsorption in the kidneys, leading to increased glucose excretion in the urine. Canagliflozin, dapagliflozin, and empagliflozin are common SGLT-2 inhibitors.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>7 - GLP-1 Receptor Agonists : </em></h6>", unsafe_allow_html=True)
                st.write('Glucagon-like peptide-1 (GLP-1) receptor agonists stimulate insulin secretion, suppress glucagon release, slow down gastric emptying, and promote satiety. They are administered as injections. Examples include exenatide, liraglutide, and dulaglutide.')

                st.markdown("<h6 style='color:  #FF7F50;'><em>8 - Insulin : </em></h6>", unsafe_allow_html=True)
                st.write('In some cases, when other medications are not sufficient to control blood sugar levels, insulin therapy may be prescribed. There are various types of insulin with different durations of action, such as rapid-acting, short-acting, intermediate-acting, and long-acting insulin.')

                st.markdown("<h4 style='color:  #FF7F50;'><em>Note : </em></h6>", unsafe_allow_html=True)
                st.write('It is essential to work closely with your healthcare provider to determine the most appropriate medication for you. The treatment plan may involve a combination of medications or adjustments over time, depending on your response to treatment and changes in your health status.')

                st.markdown("<h2 style='color: #FF5733;'><em>How to control Type 1 Diabetes without Medicines?</em></h2>", unsafe_allow_html=True)
                st.image('diabtype2withoutmedicine.jpg')

                st.markdown('For More Information of Type 2 Diabetes : [Click Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193)')








            elif diabetes_type == 'Type 3':
                st.subheader('Type 3 Diabetes :')
                list_items = ['Type 3 diabetes is a less common term that refers to a connection between insulin resistance and neurodegenerative diseases, particularly Alzheimers disease.', 'Some research suggests that insulin resistance in the brain may contribute to the development of Alzheimers disease.', 'However, Type 3 diabetes is not a widely accepted clinical classification for diabetes, and more research is needed in this area.']
                # Create an unordered list using HTML tags
                unordered_list = "<ul>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ul>"
                # Display the unordered list using st.markdown()
                st.markdown(unordered_list, unsafe_allow_html=True)

# VISUALIZATIONS
    
    
#-------------------------- -------------------------->>Heart Diseases Prediction<<----------------------------------------------



if selected == 'Heart Diseases Prediction':

    df = pd.read_csv('heart.csv')

# HEADINGS
    st.title('Heart Checkup')
    st.sidebar.header('Patient Data')
    #st.subheader('Training Data Stats')
    #st.write(df.describe())

# X AND Y DATA
    x = df.drop(['target'], axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


    def user_report():
        age = st.sidebar.slider('Age', 1, 100)
        sex = st.sidebar.slider('Sex', 0, 1)
        cp = st.sidebar.slider('Chest Pain types', 0, 3, step=1)
        trestbps = st.sidebar.slider('Resting Blood Pressure',100,200,step=1)
        chol = st.sidebar.slider('Serum Cholestoral in mg/dl', 100, 600)
        fbs = st.sidebar.slider('Fasting Blood Sugar > 120 mg/dl', 0, 1)
        restecg = st.sidebar.slider('Resting Electrocardiographic results', 0, 2, step=1)
        thalach = st.sidebar.slider('Maximum Heart Rate achieved', 50, 220)
        exang = st.sidebar.select_slider('Exercise Induced Angina', options=[0,1])
        oldpeak = st.sidebar.slider('ST depression induced by exercise', 0.0, 6.0, step=0.1)
        slope = st.sidebar.slider('Slope of the peak exercise ST segment', 0, 2, step=1)
        ca = st.sidebar.slider('Major vessels colored by fluoroscopy', 0, 4, step=1)
        thal = st.sidebar.slider('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', 0, 2, step=1)

        user_report_data = {
            'age':age,
            'sex':sex,
            'cp':cp,
            'trestbps':trestbps,
            'chol':chol,
            'fbs':fbs,
            'restecg':restecg,
            'thalach':thalach,
            'exang':exang,
            'oldpeak':oldpeak,
            'slope':slope,
            'ca':ca,
            'thal':thal

        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data
    


    # PATIENT DATA
    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)

# MODEL
    model = LogisticRegression(penalty='l2',       # L2 regularization (default)
    C=1.0,              # Regularization strength (default)
    solver='lbfgs',     # Optimization algorithm (default)
    max_iter=100,       # Maximum number of iterations (default)
    multi_class='auto', # Auto choose one-vs-rest or multinomial (default)
    class_weight=None,  # No class weights (default)
    random_state=42,    # Random seed for reproducibility
    fit_intercept=True, # Calculate the intercept (default)
    intercept_scaling=1 # No scaling of the intercept (default))
)
    model.fit(x_train, y_train)
    user_result = model.predict(user_data)


    diab_diagnosis = ''
    
    if st.button('Heart Diseases Test Result'):
        #diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if user_result[0]==0:
            diab_diagnosis = 'You are not having Heart Diseases'
        else:
            diab_diagnosis = 'You are Having Heart Diseases. Immediately Consult a Doctor'
    st.success(diab_diagnosis)
# VISUALIZATIONS
    #st.title('Visualised Patient Report')

# COLOR FUNCTION
    #if user_result[0] == 0:
            #color = 'blue'
  #  else:
#        color = 'red'

#Age vs Maximum Heart Rate

    

# OUTPUT
    
    
# creating a button for Prediction
    
    
    #st.subheader('Algorithm Used : LogisticRegression')
    #st.markdown('<h2><i class="fas fa-heartbeat"></i> Accuracy : </h2>', unsafe_allow_html=True)

    #st.subheader(str(accuracy_score(y_test, model.predict(x_test)) * 100) + '%')


if selected == 'Brain Tumor Detection':
    
    def load_model(model_path):
        model = joblib.load(model_path)
        return model

# Function to preprocess the input image for brain tumor detection
    def preprocess_image(image):
    # Resize the image to the size used during training (replace 200 with your image size)
        size = (200, 200)
        resized_image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Convert the image to grayscale (assuming the model was trained on grayscale images)
        grayscale_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2GRAY)
    # Normalize the image to values between 0 and 1
        normalized_image = grayscale_image.astype(np.float32) / 255.0
    # Return the preprocessed image
        return normalized_image

# Brain tumor detection using the pre-trained model
    def brain_tumor_detection(image, model_path):
    # Load the brain tumor detection model
        model = load_model(model_path)

    # Preprocess the image
        preprocessed_image = preprocess_image(image)

    # Reshape the preprocessed image to a format compatible with the model
        preprocessed_image = preprocessed_image.reshape(1, -1)

    # Make prediction using the loaded model
        prediction = model.predict(preprocessed_image)

    # Map prediction label to human-readable class name
        if prediction[0] == 0:
            predicted_class = 'You Have No Brain Tumor'
        else:  
            predicted_class = 'You have Positive Tumor. Immediately Consult a Doctor.'

        return predicted_class

    def main():
        st.title('Brain Tumor Detection')
        st.header('Upload an Image for Prediction')

    # File uploader for image
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
        # Read the uploaded image using PIL
            image = Image.open(uploaded_file)

        # Preprocess the image and make the prediction
            model_path = 'brain_tumor_detection_model.joblib'  # Replace with the path to your model file
            predicted_class = brain_tumor_detection(image, model_path)

        # Display the image and the prediction result
            
            #st.image(image, caption=f'Predicted Class: {predicted_class}', use_column_width=True)
            st.image(image)
            if st.button('Predict'):
                
                st.write('Predicted Class : ',predicted_class)

    if __name__ == "__main__":
        main()


#----------------------------------------------------------------------->>Pneumonia Detection<<----------------------------------------------------------------------------


if selected == 'Pneumonia Detection':

    st.title('Pneumonia Detection')

    model = tf.keras.models.load_model("Pneumonia_detection_model.h5")



# Function to make predictions
    def predict_pneumonia(model, img):
        img = cv.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        pred_probability = model.predict(img)
        # Return the probability value
        if st.button('Predict'):
            if pred_probability > 0.5:
                st.write("Prediction: You have Pneumonia. Please Consult a Doctor.")
            else:
                st.write("Prediction: You are Normal")
        return pred_probability[0][0]  

    def main():
        #st.title("Pneumonia Detection Web App")
        st.header("Upload an image for prediction.")
    
    # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
        # Read the uploaded image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            image = np.array(image)
        
        # Make prediction
            model = tf.keras.models.load_model('Pneumonia_detection_model.h5')

        # Make prediction
            st.image(image, caption="Uploaded Image", use_column_width=True)
            pred_probability = predict_pneumonia(model, image)
     
    if __name__ == "__main__":
        main()
