from pathlib import Path
import os
import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import streamlit_authenticator as stauth



radar_graph_data={}

def encode_platform(data):
    # map the platform column to integer values
    platform_map = {'udemy': -1, 'coursera': -2, 'edx': 0, 'coding_ninja': 1, 'linkedin_learning': 2,'other': 3}

    # Replace each key with its corresponding value
    for old_value, new_value in platform_map.items():
        data['certification_platform'] = data['certification_platform'].replace(old_value, str(new_value))

    # print the updated DataFrame
    return data


def encode_social_media(data):
    # map the platform column to integer values
    platform_map = {'youtube': -2, 'instagram': -1, 'facebook': 0, 'linkedin': 1, 'whatsapp': 2}

    # Replace each key with its corresponding value
    for old_value, new_value in platform_map.items():
        data['social_media'] = data['social_media'].replace(old_value, str(new_value))

    # print the updated DataFrame
    return data



def get_clean_data():
  data = pd.read_csv("data/Student_Data.csv")
  
  data = data[['cgpa','sgpa', '10th_percentage', '12th_percentage','certification_platform','social_media',
       'clubs_joined','avg_time_on_social_media']]
  
  
  return data


def add_sidebar():
    st.sidebar.header("Radar Parameter")

    data = get_clean_data()
    input_dict = {}
    for key, label in radar_graph_data.items():
      input_dict[key] = st.sidebar.slider(
                    label,
                    min_value=float(0),
                    max_value=float(data[key].max()),
                    value=float(radar_graph_data[key])
                )


    slider_labels = [
            ("cgpa", "cgpa"),
            ("sgpa", "sgpa"),
            ("10th_percentage", "10th_percentage"),
            ("12th_percentage", "12th_percentage"),
            ("clubs_joined", "clubs_joined"),
            ("avg_time_on_social_media", "avg_time_on_social_media"),
        ]

    certificate_platforms = ["Coursera", "edX", "Udemy", "Coding_Ninja", "Linkedin_Learning"]
    social_media_platforms = ["LinkedIn", "WhatsApp", "Instagram","Facebook" , "Youtube"]

    for label, key in slider_labels:
      if key in data.columns:
        input_dict[key] = st.sidebar.slider(
                    label,
                    min_value=float(0),
                    max_value=float(data[key].max()),
                    value=float(data[key].mean())
                )
      else:
        st.sidebar.warning(f"Column {key} not found in the DataFrame.")

    selected_certificate_platform = st.sidebar.selectbox("Select Certificate Platform", certificate_platforms)

    selected_social_media_platform = st.sidebar.selectbox("Select Social Media Platform", social_media_platforms)

    input_dict['social_media'] = selected_social_media_platform.lower()
    input_dict['certification_platform'] = selected_certificate_platform.lower()

    return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    if key:
      max_val = X[key].max()
      min_val = X[key].min()
      scaled_value = (value - min_val) / (max_val - min_val)
      scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
    
    diagram_input_data=input_data.copy()
    diagram_input_data.pop('social_media')
    diagram_input_data.pop('certification_platform')
    diagram_input_data= get_scaled_values(diagram_input_data) 
    
    categories = ['cgpa', 'sgpa', '10th_percentage', '12th_percentage',
                  'clubs_joined', 'avg_time_on_social_media']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            diagram_input_data['cgpa'], diagram_input_data['sgpa'], diagram_input_data['10th_percentage'],
            diagram_input_data['12th_percentage'],diagram_input_data['clubs_joined'], diagram_input_data['avg_time_on_social_media']
        ],
        theta=categories,
        fill='toself',
        name='performance'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
  model = pickle.load(open("model/model_xgb.pkl", "rb"))
  scaler = pickle.load(open("model/scaler_xgb.pkl", "rb"))
  
  input_data=encode_platform(input_data)
  input_data=encode_social_media(input_data)
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Student performance prediction")
  st.write("The Student is:")
  
  if prediction[0] == 1:
    st.write("<span class='diagnosis benign'>Fast Learner</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Slow Learner</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being Slow learner: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being Fast learner: ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("This app can assist Teachers in making a academic analysis, but should not be used as a substitute for a professional.")



import streamlit as st
import streamlit_authenticator as stauth
from dependancies import sign_up, fetch_users


# st.set_page_config(page_title='Streamlit', page_icon='🐍', initial_sidebar_state='collapsed')

def main():
                  
  
    
  st.set_page_config(page_title="LTA",page_icon=":bar_chart:",layout="wide")
            
            
  try:
      users = fetch_users()
      emails = []
      usernames = []
      passwords = []
      gfm=[]

      for user in users:
          emails.append(user['key'])
          usernames.append(user['username'])
          passwords.append(user['password'])
          gfm.append('gfm')

      credentials = {'usernames': {}}
      for index in range(len(emails)):
          credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}

      Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)

      email, authentication_status, username = Authenticator.login(':green[Login]', 'main')

      info, info1 = st.columns(2)

      if not authentication_status:
          sign_up()

      if username:
          if username in usernames:
              if authentication_status:
                  st.sidebar.subheader(f'Welcome {username}')
                  Authenticator.logout('Log Out', 'sidebar')
                  with open("assets/style.css") as f:
                    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
                  
                  #authenticator.logout("Logout","sidebar")
                  input_data = add_sidebar()
                  
                  with st.container():
                    # st.header("Learn to Aanlyze(LTA)")
                    st.markdown("""
                    <h1 style='text-align: center; font-weight: bold;'>Learn to Aanlyze(LTA)</h1>
                """, unsafe_allow_html=True)

                    st.write("Welcome {}".format(username))
                    
                    if gfm=="Yes":
                      uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

                      if uploaded_file is not None:
                        # Display details about the uploaded file
                        st.write("File Details:")
                        st.write(f"Filename: {uploaded_file.name}")
                        st.write(f"File Type: {uploaded_file.type}")
                        st.write(f"File Size: {uploaded_file.size} bytes")

                        # Save the uploaded file to a folder
                        save_folder = "E:\Test code\data"
                        os.makedirs(save_folder, exist_ok=True)

                        # Check if uploaded_file has a name attribute
                        if hasattr(uploaded_file, 'name') and uploaded_file.name is not None:
                            save_path = os.path.join(save_folder, uploaded_file.name)
                            with open(save_path, "wb") as file:
                                file.write(uploaded_file.getvalue())

                            st.success(f"File saved successfully to: {save_path}")
                  
                                  
                    
                    
                    
                    
                    
                    
                    
                    with st.form(key='searchform'):
                      dropdown_options = ['Third Year', 'Fourth Year']
                      nav1,nav2,nav3,nav4=st.columns([3,1,1,1])

                      with nav1:
                        rbt_number=st.text_input("Enter RBT Number:")  
                      # with nav2:
                      #   roll_number=st.text_input("Enter Roll Number: ")
                      with nav3:
                        selected_year = st.selectbox('Select Year:', dropdown_options)
                      with nav4:
                        st.write("")
                        st.write("")
                        submit_button = st.form_submit_button(label='Search')
                        
                    main_file="data/Student_Data.csv"
                    radar_graph_data={}
                    count=0
                    data=pd.read_csv(main_file)
                    second_year_3rd_sem={}
                    second_year_4th_sem={}
                    third_year_5th_sem={}
                    third_year_6th_sem={}
                    desired_column = data['prn_number'].tolist()
                    if rbt_number in desired_column:
                      if rbt_number and selected_year=='Third Year':

                        matched_row_4th_year={}
                        radar_graph_data={}
                        csv_path = "data/Students_marks_in_subjects(ty).csv"  # Replace with the actual path to your CSV file
                        df = pd.read_csv(csv_path)
                        # Case-insensitive search on all columns
                        filtered_rows = df[df.apply(lambda row: any(str(cell).lower().startswith(rbt_number.lower()) for cell in row), axis=1)]

                        radar_filtered_rows=data[data.apply(lambda row: any(str(cell).lower().startswith(rbt_number.lower()) for cell in row), axis=1)]
                        
                        if len(radar_filtered_rows) == 1:
                          matched_row = radar_filtered_rows.iloc[0]

                          # Specify the columns you want to access
                          selected_columns = ["cgpa", "sgpa", "10th_percentage", "12th_percentage","clubs_joined","avg_time_on_social_media","certification_platform","social_media"]

                          # Getting the selected values from the matched row
                          for column in selected_columns:
                            radar_graph_data[column] = matched_row[column]
                            
                          df = pd.DataFrame(list(radar_graph_data.items()), columns=['Main Parameters', 'Value'])

                          # Displaying key-value pairs in a table without row and column numbers
                          st.write(df)
                        
                        st.write("Overall Academic record of the Student:")
                        st.write(filtered_rows)
                        
                        matched_row_3rd_year= filtered_rows.iloc[0]
                        
                        columns_3rd_sem=["FLAT","OOP","COA","SE"]
                        second_year_3rd_sem={}
                        for column in columns_3rd_sem:
                            second_year_3rd_sem[column] = int(matched_row_3rd_year[column]) 
                      
                        columns_4th_sem=["OS","DBMS","CT","CG","STQA"]
                        second_year_4th_sem={}
                        for column in columns_4th_sem:
                            second_year_4th_sem[column] = int(matched_row_3rd_year[column]) 
                        count=1
                      
                      if rbt_number and selected_year=='Fourth Year':
                        matched_row_3rd_year={}
                        radar_graph_data={}
                        csv_path = "data/Students_marks_in_subjects(be).csv"  # Replace with the actual path to your CSV file
                        df = pd.read_csv(csv_path)
                        # Case-insensitive search on all columns
                        filtered_rows = df[df.apply(lambda row: any(str(cell).lower().startswith(rbt_number.lower()) for cell in row), axis=1)]

                        radar_filtered_rows=data[data.apply(lambda row: any(str(cell).lower().startswith(rbt_number.lower()) for cell in row), axis=1)]
                        
                        if len(radar_filtered_rows) == 1:
                          matched_row = radar_filtered_rows.iloc[0]

                          # Specify the columns you want to access
                          selected_columns = ["cgpa", "sgpa", "10th_percentage", "12th_percentage","clubs_joined","avg_time_on_social_media","certification_platform","social_media"]

                          # Getting the selected values from the matched row
                          for column in selected_columns:
                            radar_graph_data[column] = matched_row[column]
                            
                          df = pd.DataFrame(list(radar_graph_data.items()), columns=['Main Parameters', 'Value'])

                          # Displaying key-value pairs in a table without row and column numbers
                          st.write(df)
                        
                        st.write("Overall Academic record of the Student:")
                        st.write(filtered_rows)
                        matched_row_4th_year= filtered_rows.iloc[0]
                        
                        columns_3rd_sem=["FLAT","OOP","COA","SE"]
                        second_year_3rd_sem={}
                        for column in columns_3rd_sem:
                            second_year_3rd_sem[column] = int(matched_row_4th_year[column]) 
                      
                        columns_4th_sem=["OS","DBMS","CT","CG","STQA"]
                        second_year_4th_sem={}
                        for column in columns_4th_sem:
                            second_year_4th_sem[column] = int(matched_row_4th_year[column]) 
                        count=1
                        
                        columns_5th_sem=["AI","CN","DAA","ML/CNS","FMSF"]
                        third_year_5th_sem={}
                        for column in columns_5th_sem:
                            third_year_5th_sem[column] = int(matched_row_4th_year[column]) 
                      
                        columns_6th_sem=["CMA","IOT","CD","DM/IS","MWA"]
                        third_year_6th_sem={}
                        for column in columns_6th_sem:
                            third_year_6th_sem[column] = int(matched_row_4th_year[column]) 
                        count=1
                      
                      
                      if count==1 and len(filtered_rows):
                        st.success("The student with {} found".format(rbt_number))
                      else:
                        st.error("The student with {} not found".format(rbt_number))
                    else:
                      st.error("The student with {} not found".format(rbt_number)) 
                      
                      
                          
                    with st.container():
                      st.title("Student Academic Performance")
                      st.write("The Student Academic Performance Predictor is an intelligent system designed to forecast and evaluate a student's academic success based on a comprehensive set of relevant factors. Leveraging advanced machine learning algorithms, this tool aims to assist educators, administrators, and students themselves in understanding and optimizing the pathways to academic achievement. ")
                    
                    with st.container():
                      col1, col2 = st.columns([4,1])
                      
                      with col1:
                        content="Radar Chart of the Student"
                        st.markdown(f'<div style="text-align: center; font-weight: bold; font-size: 20px;">{content}</div>', unsafe_allow_html=True)
                        radar_chart1 = get_radar_chart(input_data)
                        st.plotly_chart(radar_chart1)
                        
                      with col2:
                        # if radar_graph_data:
                        #   input_data=radar_graph_data
                        add_predictions(input_data)
                      
                    with st.container():
                      show_new_data=0
                      if second_year_3rd_sem and second_year_4th_sem:
                        st.markdown("""
                      <h3 style='text-align: center; font-weight: bold;'>Marks gained in Second Year</h3>
                  """, unsafe_allow_html=True)

                        chart_data = pd.DataFrame(list(second_year_3rd_sem.items()), columns=["Category", "Value"])
                        # Display the DataFrame
                        st.write("Marks gained in 3rd Semester:")
                        # Create a bar chart using the DataFrame
                        st.bar_chart(chart_data.set_index("Category"))
                        
                        chart_data = pd.DataFrame(list(second_year_4th_sem.items()), columns=["Category", "Value"])
                        # Display the DataFrame
                        st.write("Marks gained in 4th Semester:")
                        # Create a bar chart using the DataFrame
                        st.bar_chart(chart_data.set_index("Category"))
                        show_new_data=1


                      if show_new_data and third_year_5th_sem and third_year_6th_sem:
                        st.markdown("""
                      <h3 style='text-align: center; font-weight: bold;'>Marks gained in Third Year</h3>
                  """, unsafe_allow_html=True)
                        
                        chart_data = pd.DataFrame(list(third_year_5th_sem.items()), columns=["Category", "Value"])
                        # Display the DataFrame
                        st.write("Marks gained in 5th Semester:")
                        # Create a bar chart using the DataFrame
                        st.bar_chart(chart_data.set_index("Category"))
                        
                        chart_data = pd.DataFrame(list(third_year_6th_sem.items()), columns=["Category", "Value"])
                        # Display the DataFrame
                        st.write("Marks gained in 6th Semester:")
                        # Create a bar chart using the DataFrame
                        st.bar_chart(chart_data.set_index("Category"))

              elif not authentication_status:
                with info:
                    st.error('Incorrect Password or username')
              else:
                with info:
                    st.warning('Please feed in your credentials')
          else:
            with info:
                st.warning('Username does not exist, Please Sign up')


  except:
    st.success('Refresh Page')

 
if __name__ == '__main__':
  main()