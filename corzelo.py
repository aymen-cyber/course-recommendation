import numpy as np
import streamlit as st
import spacy
import NoWhere as nh
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from math import sqrt, pow, exp
from streamlit_option_menu import option_menu
import spacy
import streamlit as st
import pandas as pd
import base64,random
import time,datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io,random
from streamlit_tags import st_tags
from PIL import Image
#import pymysql
from course import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
#import pafy
import plotly.express as px
import pyresparser

st.set_page_config(
  page_title="project Pfe",
  page_icon=None,
  layout="wide",
  
)

nlp = spacy.load("en_core_web_sm")

st.title("Project Validation")

expander = st.expander("About this app", expanded=True)
with expander:
    st.info("""
    The **COURZELO** *course recom* app is an easy-to-use interface built in Streamlit using many similarity algorithm!
    This app is the first of 3 parts  !""")

def streamlit_menu():
  selected = option_menu(
    menu_title=None, 
    options = ["Cours Recomondation","CV Scraping","Dashborad"],
    orientation = "horizontal"
  )
  return selected

selected = streamlit_menu()

if selected == "Cours Recomondation":
  dataset_btn = st.sidebar.button("Show Dataset")

  text_input = st.sidebar.text_input("Search For A Course", key="1")
  recnum = st.sidebar.slider("Pick the number of courses you want", min_value=1, max_value=50, value=20, step=1)


  genre = st.sidebar.radio(
      "What's your favorite similarity measures ?",
      ('Cosine Similarity', 'Euclidean Distance'))

  is_in_the_way = st.checkbox ("Show Recommendation Result ")

#   if is_in_the_way:
#     if text_input:
#       if genre == 'Cosine Similarity':
#         try:  
#             cosres = nh.recommend_course(text_input, recnum)
#         except:
#             cosres = ("not found",1)
#         cosres= cosres.values
#         cosres = cosres.tolist()
#         st.write('## This is a simple implementation of `Cosine similarity` using `Courses Dataset`')
#         for i in cosres:
#           st.markdown(f'* {i[0]}')
  if is_in_the_way:
    if text_input:
      if genre == 'Cosine Similarity':
            try:  
                cosres = nh.recommend_course(text_input, recnum)
                cosres = cosres.values
                cosres = cosres.tolist()
                st.write('## This is a simple implementation of `Cosine similarity` using `Courses Dataset`')
                for i in cosres:
                    st.markdown(f'* {i[0]}')
            except:
                cosres = [("not found", 1)]
                st.write("Course not found.")
      else:
        if genre == 'Euclidean Distance':
          cosres = nh.recommend_course_euc(text_input, recnum)
          cosres= cosres.values
          cosres = cosres.tolist()
          st.write('## This is a simple implementation of `Euclidean Distance` using `Courses Dataset`')
          for i in cosres:
            st.markdown(f'* {i[0]}')


  jaccinp = []
  jaccinp0 = st.sidebar.text_input("Enter fst jacc data", key="2")
  jaccinp1 = st.sidebar.text_input("Enter scd jacc data", key="3")


  X =jaccinp0.lower()
  Y =jaccinp1.lower()
    
  # tokenization
  X_list = word_tokenize(X) 
  Y_list = word_tokenize(Y)
    
  # sw contains the list of stopwords
  sw = stopwords.words('english') 
  l1 =[];l2 =[]
    
  # remove stop words from the string
  X_set = {w for w in X_list if not w in sw} 
  Y_set = {w for w in Y_list if not w in sw}
    

  resultjacc = st.sidebar.button("Calculate Jaccard Distance")
  resultcos = st.sidebar.button("Calculate Cosine similarity")
  resulteuc = st.sidebar.button("Calculate Euclidean Distance")
  jaccinp.append(jaccinp0)
  jaccinp.append(jaccinp1)


  if resultjacc:
    # text_input = False
    # genre = False
    jaccinp = [sent.lower().split(" ") for sent in jaccinp]
    jaccres = nh.jaccard_similarity(jaccinp[0], jaccinp[1])
    st.info(f'## the `similarity` between the two sentences using `Jaccard Distance` is : `{jaccres}` ')

  if resultcos:
    # text_input = False
    # genre = False
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
      if w in X_set: l1.append(1) # create a vector
      else: l1.append(0)
      if w in Y_set: l2.append(1)
      else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
      c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    st.info(f'## the `similarity` between the two sentences using `Cosine similarity` is : `{cosine}`')

  if resulteuc:
    # text_input = False
    # genre = False
    jaccinp = [sent.lower().split(" ") for sent in jaccinp]
    embeddings = [nlp(str(sentence)).vector for sentence in jaccinp]
    distance = nh.euclidean_distance(embeddings[0], embeddings[1])
    st.info(f'## the `similarity` between the two sentences using `Euclidean Distance` is : `{nh.distance_to_similarity(distance)}`')


  if dataset_btn:
    st.table(nh.fdf())

if selected == "CV Scraping":
  

  # def fetch_yt_video(link):
  #     video = pafy.new(link)
  #     return video.title

  def get_table_download_link(df,filename,text):
      """Generates a link allowing the data in a given panda dataframe to be downloaded
      in:  dataframe
      out: href string
      """
      csv = df.to_csv(index=False)
      b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
      # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
      href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
      return href

  def pdf_reader(file):
      resource_manager = PDFResourceManager()
      fake_file_handle = io.StringIO()
      converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
      page_interpreter = PDFPageInterpreter(resource_manager, converter)
      with open(file, 'rb') as fh:
          for page in PDFPage.get_pages(fh,
                                        caching=True,
                                        check_extractable=True):
              page_interpreter.process_page(page)
              print(page)
          text = fake_file_handle.getvalue()

      # close open handles
      converter.close()
      fake_file_handle.close()
      return text

  def show_pdf(file_path):
      with open(file_path, "rb") as f:
          base64_pdf = base64.b64encode(f.read()).decode('utf-8')
      pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
      #pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)

  def course_recommender(course_list):
      st.subheader("**Courses & Certificates🎓 Recommendations**")
      c = 0
      rec_course = []
      no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
      random.shuffle(course_list)
      for c_name, c_link in course_list:
          c += 1
          st.markdown(f"({c}) [{c_name}]({c_link})")
          rec_course.append(c_name)
          if c == no_of_reco:
              break
      return rec_course

  # connection = pymysql.connect(host='localhost',user='root',password='',db='sra')
  # cursor = connection.cursor()

  # def insert_data(name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses):
  #     DB_table_name = 'user_data'
  #     insert_sql = "insert into " + DB_table_name + """
  #     values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
  #     rec_values = (name, email, str(res_score), timestamp,str(no_of_pages), reco_field, cand_level, skills,recommended_skills,courses)
  #     cursor.execute(insert_sql, rec_values)
  #     connection.commit()

  #def run():
  if 1 == 1:
    st.title("Smart Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    # link = '[©Developed by Spidy20](http://github.com/spidy20)'
    # st.sidebar.markdown(link, unsafe_allow_html=True)
    #img = Image.open('SRA_Logo.jpg')
    #img = img.resize((250,250))
    #st.image(img)

    # Create the DB
    # db_sql = """CREATE DATABASE IF NOT EXISTS SRA;"""
    # cursor.execute(db_sql)

    # # Create table
    # DB_table_name = 'user_data'
    # table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
    #                 (ID INT NOT NULL AUTO_INCREMENT,
    #                  Name varchar(100) NOT NULL,
    #                  Email_ID VARCHAR(50) NOT NULL,
    #                  resume_score VARCHAR(8) NOT NULL,
    #                  Timestamp VARCHAR(50) NOT NULL,
    #                  Page_no VARCHAR(5) NOT NULL,
    #                  Predicted_Field VARCHAR(25) NOT NULL,
    #                  User_level VARCHAR(30) NOT NULL,
    #                  Actual_skills VARCHAR(300) NOT NULL,
    #                  Recommended_skills VARCHAR(300) NOT NULL,
    #                  Recommended_courses VARCHAR(600) NOT NULL,
    #                  PRIMARY KEY (ID));
    #                 """
    # cursor.execute(table_sql)
    if choice == 'Normal User':
        # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Upload your resume, and get smart recommendation based on it."</h4>''',
        #             unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            # with st.spinner('Uploading your Resume....'):
            #     time.sleep(4)
            save_image_path = pdf_file.name
            
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            #resume_data = ResumeParser(save_image_path, config_file_path="./config.cfg").get_extracted_data()
            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >=3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
          
                st.subheader("**Skills Recommendation💡**")
                
                ## Skill shows
                keywords = st_tags(label='### Skills that you have',
                text='See our skills recommendation',
                    value=resume_data['skills'],key = '1')

                ##  recommendation
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                            'javascript', 'angular js', 'c#', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                ## Courses recommendation
                for i in resume_data['skills']:
                    ## Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break

                    ## Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        break

                    ## Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break

                    ## IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break

                    ## Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break

                #
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                if 'Objective' in resume_text:
                    resume_score = resume_score+20
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Declaration'  in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration✍/h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',unsafe_allow_html=True)

                if 'Hobbies' or 'Interests'in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                if 'Achievements' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅 </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Achievements🏅. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

                st.subheader("**Resume Score📝**")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score is calculated based on the content that you have added in your Resume. **")
                st.balloons()
                """
                insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                            str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                            str(recommended_skills), str(rec_course))
                

                ## Resume writing video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                #res_vid_title = fetch_yt_video(resume_vid)
                #st.subheader("✅ **"+res_vid_title+"**")
                #st.video(resume_vid)

                ## Interview Preparation Video
                st.header("**Bonus Video for Interview👨‍💼 Tips💡**")
                interview_vid = random.choice(interview_videos)
                int_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ **" + int_vid_title + "**")
                st.video(interview_vid)

                connection.commit()"""
            else:
                st.error('Something went wrong..')
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'machine_learning_hub' and ad_password == 'mlhub123':
                st.success("Welcome Kushal")
                # Display Data
                # cursor.execute('''SELECT*FROM user_data''')
                data = ['1', 'aymen', 'aymen.khazri@gmail.com', '94', '17', '2','data_science', 'advanced', 'python', 'js','ds_course']
                data = [data,data,data,data,data,data,data,data,data,data,data]
                st.header("**User's👨‍💻 Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                'Recommended Course'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                # query = 'select * from user_data;'
                Predicted_Field = data
                plot_data = [data,data,data,data,data,data,Predicted_Field,data,data,data,data]
                
                ## Pie chart for predicted field recommendations
                #labels = plot_data.Predicted_Field.unique()
                labels = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                print(labels)
                #values = plot_data.Predicted_Field.value_counts()
                values = [2,6,9,8,7,4,5,6,8,1]
                print(values)
                st.subheader("📈 **Pie-Chart for Predicted Field Recommendations**")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                ### Pie chart for User's👨‍💻 Experienced Level
                #labels = plot_data.User_level.unique()
                labels = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                #values = plot_data.User_level.value_counts()
                values = [2,6,9,8,7,4,5,6,8,1]
                st.subheader("📈 ** Pie-Chart for User's👨‍💻 Experienced Level**")
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                st.plotly_chart(fig)


            else:
                st.error("Wrong ID & Password Provided")
  #run()

if selected == "Dashborad":
  pass
