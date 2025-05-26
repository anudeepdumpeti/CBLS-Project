from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
import pymysql
import os
from django.core.files.storage import FileSystemStorage
from datetime import datetime
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import smtplib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


global uname, graph
dataset = pd.read_csv("model/dataset.csv")
labels = np.unique(dataset['FinalResult'])
le = LabelEncoder()
dataset['FinalResult'] = pd.Series(le.fit_transform(dataset['FinalResult'].astype(str)))#encode all str columns to numeric

Y = dataset['FinalResult'].ravel()
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
print(X)
sc = MinMaxScaler()
X = sc.fit_transform(X)
print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global graph
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') *100
    r = recall_score(y_test, predict,average='macro') *100
    f = f1_score(y_test, predict,average='macro') *100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    '''
    conf_matrix = confusion_matrix(y_test, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.title("Teacher's Effectiveness Based on Student Performance Grades")
    plt.close()
    graph = base64.b64encode(buf.getvalue()).decode()
    '''
    
nb_cls = GaussianNB()
nb_cls.fit(X_train, y_train)
predict = nb_cls.predict(X_test)
calculateMetrics("Naive Bayes", y_test, predict)

xg_cls = XGBClassifier()
xg_cls.fit(X_train, y_train)
predict = xg_cls.predict(X_test)
calculateMetrics("XGBoost", y_test, predict)

def generateMetricsGraph():
    global accuracy, precision, recall, fscore
    algorithms = ['Naive Bayes', 'XGBoost']
    
    # Create a bar chart for the metrics
    x = np.arange(len(algorithms))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy', color='skyblue')
    ax.bar(x - 0.5 * width, precision, width, label='Precision', color='orange')
    ax.bar(x + 0.5 * width, recall, width, label='Recall', color='green')
    ax.bar(x + 1.5 * width, fscore, width, label='F1-Score', color='red')

    # Add labels, title, and legend
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Metrics by Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    graph = base64.b64encode(buf.getvalue()).decode()
    return graph

def getClassMembers(faculty, subject):
    count = 0
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select count(faculty_name) from choosefaculty where faculty_name='"+faculty+"' and class_name='"+subject+"'")
        rows = cur.fetchall()
        for row in rows:
            count = row[0]
            break
    return count

def getSubject(faculty):
    subject = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select teaching_subjects from faculty where username='"+faculty+"'")
        rows = cur.fetchall()
        for row in rows:
            subject = row[0]
            break
    return subject

def AutoFacultySelection(request):
    if request.method == 'GET':
        global uname
        faculty = request.GET.get('name', False)
        subject = request.GET.get('subject', False)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        members_count = getClassMembers(faculty, subject)
        status = "Slot already crossed with 60 members. You can try other faculty"        
        if members_count <= 60:
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO choosefaculty VALUES('"+faculty+"','"+uname+"','"+subject+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "You are successfully assigned to faculty "+faculty
        context= {'data': status}
        return render(request, 'StudentScreen.html', context)

def ChooseFaculty(request):
    if request.method == 'GET':
        # Create the table head using modern styling
        output = '''
        <div class="data-table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Faculty Name</th>
                        <th>Feedback</th>
                        <th>Ratings</th>
                        <th>Subject Name</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        scores = []
        labels = []
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        
        with con:
            cur = con.cursor()
            cur.execute("select * from feedback")
            rows = cur.fetchall()
            for row in rows:
                subject = getSubject(row[1])
                # Create each row with modern styling and icons
                output += f'''
                <tr>
                    <td>{row[1]}</td>
                    <td>{str(row[2])}</td>
                    <td><div class="rating-stars">{generate_rating_stars(row[3])}</div></td>
                    <td>{subject}</td>
                    <td>
                        <a href="AutoFacultySelection?name={str(row[1])}&subject={subject}" class="action-button">
                            <i class="fas fa-user-check"></i> Select
                        </a>
                    </td>
                </tr>
                '''
        
        output += '''
                </tbody>
            </table>
        </div>
        
        <style>
            .data-table-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
                overflow: hidden;
                margin-bottom: 2rem;
            }
            
            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Montserrat', sans-serif;
            }
            
            .data-table th {
                background: #f1f5f9;
                color: #1e293b;
                font-weight: 600;
                text-align: left;
                padding: 1rem;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #e2e8f0;
            }
            
            .data-table td {
                padding: 1rem;
                color: #334155;
                border-bottom: 1px solid #e2e8f0;
                font-size: 0.95rem;
                vertical-align: middle;
            }
            
            .data-table tr:last-child td {
                border-bottom: none;
            }
            
            .data-table tr:hover {
                background: #f8fafc;
            }
            
            .rating-stars {
                color: #f59e0b;
                letter-spacing: 2px;
            }
            
            .action-button {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: #2563eb;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .action-button:hover {
                background: #1d4ed8;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(37, 99, 235, 0.2);
            }
            
            @media (max-width: 768px) {
                .data-table-container {
                    border-radius: 8px;
                    overflow-x: auto;
                }
                
                .data-table th,
                .data-table td {
                    padding: 0.75rem;
                }
                
                .action-button {
                    padding: 0.4rem 0.8rem;
                }
            }
        </style>
        '''
        
        context = {'data': output}
        return render(request, 'FacultySuggestion.html', context)
        
# Helper function to generate star ratings
def generate_rating_stars(rating):
    rating = float(rating)
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    stars = '<i class="fas fa-star"></i>' * full_stars
    if half_star:
        stars += '<i class="fas fa-star-half-alt"></i>'
    stars += '<i class="far fa-star"></i>' * empty_stars
    
    return stars    

def Feedback(request):
    if request.method == 'GET':
        output = '<div class="form-group">'
        output += '<label class="form-label">Choose Faculty</label>'
        output += '<select name="t1" class="form-control">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from faculty")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select>"
        output += "</div>"
        context= {'data1': output}
        return render(request, 'Feedback.html', context)

def FeedbackAction(request):
    if request.method == 'POST':
        global uname
        faculty = request.POST.get('t1', False)
        feedback = request.POST.get('t2', False)
        rating = request.POST.get('t3', False)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO feedback VALUES('"+uname+"','"+faculty+"','"+feedback+"','"+rating+"','"+current_datetime+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        status = "error in submitting feedback"
        if db_cursor.rowcount == 1:
            status = "Your feedback successfully submitted"
        context= {'data': status}
        return render(request, 'StudentScreen.html', context)    

def FacultySuggestion(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Faculty Name</th><th><font size="3" color="black">Subject Name</th>'
        output+='<th><font size="3" color="black">Performance %</th><th><font size="3" color="black">Choose Faculty</th></tr>'
        scores = []
        labels = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select faculty_name, avg(subject_marks) from marks group by faculty_name")
            rows = cur.fetchall()
            for row in rows:
                subject = getSubject(row[0])
                scores.append(row[1])
                labels.append(row[0])
                output+='<tr><td><font size="3" color="black">'+row[0]+'</td><td><font size="3" color="black">'+subject+'</td>'
                output += '<td><font size="3" color="black">'+str(row[1])+'</td>'
                output+='<td><a href=\'AutoFacultySelection?name='+str(row[0])+'&subject='+subject+'\'><font size=3 color=blue>Choose Faculty</font></a></td></tr>' 
        output+= "</table></br></br></br></br>" 
        context= {'data':output}
        return render(request, 'FacultySuggestion.html', context)

def runML(marks):
    global xg_cls, sc, labels
    data = []
    data.append([marks])
    data = np.asarray(data)
    data = sc.transform(data)
    predict = xg_cls.predict(data)[0]
    return labels[predict]

def ViewMarks(request):
    if request.method == 'GET':
       return render(request, 'ViewMarks.html', {}) 

def ViewMarksAction(request):
    if request.method == 'POST':
        global uname
        course = request.POST.get('t1', False)
        year = request.POST.get('t2', False)
        total = 0
        count = 0
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Subject Name</th><th><font size="" color="black">Obtained Marks</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select subject_name,subject_marks from marks where student_name='"+uname+"' and course_year='"+year+"' and course_name='"+course+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td></tr>'
                total += row[1]
                count += 1
        output+='<tr><td>-</td><td>-</td></tr><tr><td><font size="" color="black">Total Marks</td><td><font size="" color="black">'+str(total)+'</td></tr>'
        output+='<tr><td><font size="" color="black">Average GPA</td><td><font size="" color="black">'+str(total/count)+'</td></tr>'
        feedback_ml = runML(total/count)
        output+='<tr><td><font size="" color="black">ML Predicted Feedback</td><td><font size="" color="black">'+feedback_ml+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'StudentScreen.html', context)    


def ViewMessages(request):
    if request.method == 'GET':
        global uname
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Sender Name</th><th><font size="" color="black">Subject</th>'
        output+='<th><font size="" color="black">Message</th><th><font size="" color="black">Message Date</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            #cur.execute("select * from messages where receiver_name='"+uname+"'')
            cur.execute("SELECT * FROM messages WHERE receiver_name=%s", (uname,))

            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+str(row[4])+'</td></tr>'                
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'StudentScreen.html', context) 

def DownloadMaterialAction(request):
    if request.method == 'GET':
        filename = request.GET.get('name', False)
        with open("StudentApp/static/files/"+filename, "rb") as file:
            content = file.read()
        file.close()
        response = HttpResponse(content,content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename='+filename
        return response

def DownloadMaterials(request):
    if request.method == 'GET':
        global uname
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Material Name</th>'
        output+='<th><font size="" color="black">Description</th><th><font size="" color="black">Filename</th>'
        output+='<th><font size="" color="black">Upload Date</th><th><font size="" color="black">Click Here to Download</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from uploadmaterial")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+str(row[3])+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td>'
                output+='<td><a href=\'DownloadMaterialAction?name='+str(row[3])+'\'><font size=3 color=black>Download</font></a></td></tr>' 
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'StudentScreen.html', context) 

def StudentMessagingAction(request):
    if request.method == 'POST':
        global uname
        fname = request.POST.get('sname', False)
        subject = request.POST.get('t1', False)
        message = request.POST.get('t2', False)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO messages VALUES('"+uname+"','"+fname+"','"+subject+"','"+message+"','"+current_datetime+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Message successfully sent to teacher "+fname
        context= {'data': status}
        return render(request, 'StudentScreen.html', context)

def StudentMessaging(request):
    if request.method == 'GET':
        global uname
        output = '<div class="form-group"><label class="form-label">Choose Teacher Name</label>'
        output += '<select name="sname" class="form-control">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from faculty")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></div>"
        context= {'data1': output}
        return render(request, 'StudentMessaging.html', context)

def ViewAssignments(request):
    if request.method == 'GET':
        global uname
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Course Name</th>'
        output+='<th><font size="" color="black">Subject Name</th><th><font size="" color="black">Course Year</th>'
        output+='<th><font size="" color="black">Assignment Task</th><th><font size="" color="black">Description</th>'
        output+='<th><font size="" color="black">Assignment Date</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from assignments")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+str(row[3])+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td><td><font size="" color="black">'+str(row[5])+'</td>'
                output+='<td><font size="" color="black">'+row[6]+'</td></tr>' 
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'StudentScreen.html', context) 

def ViewStudentMessages(request):
    if request.method == 'GET':
        global uname
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Sender Name</th><th><font size="" color="black">Subject</th>'
        output+='<th><font size="" color="black">Message</th><th><font size="" color="black">Message Date</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from messages where receiver_name='"+uname+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+str(row[4])+'</td></tr>'                
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'FacultyScreen.html', context)

def ViewProgressReportAction(request):
    if request.method == 'POST':
        global uname
        sname = request.POST.get('sname', False)
        course = request.POST.get('t1', False)
        year = request.POST.get('t2', False)
        total = 0
        count = 0
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Subject Name</th><th><font size="" color="black">Obtained Marks</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select subject_name,subject_marks from marks where student_name='"+sname+"' and course_year='"+year+"' and course_name='"+course+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td></tr>'
                total += row[1]
                count += 1
        output+='<tr><td>-</td><td>-</td></tr><tr><td><font size="" color="black">Total Marks</td><td><font size="" color="black">'+str(total)+'</td></tr>'
        output+='<tr><td><font size="" color="black">Average GPA</td><td><font size="" color="black">'+str(total/count)+'</td></tr>'
        feedback_ml = runML(total/count)
        output+='<tr><td><font size="" color="black">ML Predicted Feedback</td><td><font size="" color="black">'+feedback_ml+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'FacultyScreen.html', context)    

def ViewProgressReport(request):
    if request.method == 'GET':
        output = '<div class="form-group"><label class="form-label">Choose Student Name</label>'
        output += '<select name="sname" class="form-control">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></div>"
        context= {'data1': output}
        return render(request, 'ViewProgressReport.html', context)

def sendMail(subject, msg, email):
    print("sending reminder to mail")
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'
        email_password = 'xyljzncebdxcubjq'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="kaleem202120@gmail.com", to_addrs=em, msg="Subject : "+subject+"\n"+msg)   

def getEmail(sname):
    email = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select email from student where username='"+sname+"'")
        rows = cur.fetchall()
        for row in rows:
            email = row[0]
            break
    return email    

def MessagingAction(request):
    if request.method == 'POST':
        global uname
        status = "error in sending message"
        sname = request.POST.get('sname', False)
        subject = request.POST.get('t1', False)
        message = request.POST.get('t2', False)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO messages VALUES('"+uname+"','"+sname+"','"+subject+"','"+message+"','"+current_datetime+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        email = getEmail(sname)
        sendMail(subject, message, email)
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Message successfully sent to "+sname+"<br/>Email sent to Parent Email : "+email
        context= {'data': status}
        return render(request, 'FacultyScreen.html', context)

def Messaging(request):
    if request.method == 'GET':
        output = '<div class="form-group"><label class="form-label">Choose Student Name</label>'
        output += '<select name="sname" class="form-control">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></div>"
        context= {'data1': output}
        return render(request, 'Messaging.html', context)

def AddMarksAction(request):
    if request.method == 'POST':
        global uname
        status= "error in adding marks details"
        sname = request.POST.get('sname', False)
        course = request.POST.get('t1', False)
        subject = request.POST.get('t2', False)
        year = request.POST.get('t3', False)
        marks = request.POST.get('t4', False)
        feedback = request.POST.get('t5', False)
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO marks VALUES('"+sname+"','"+uname+"','"+course+"','"+year+"','"+subject+"','"+marks+"','"+feedback+"','"+current_datetime+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Marks details successfully submitted"
        context= {'data': status}
        return render(request, 'FacultyScreen.html', context)

def AddMarks(request):
    if request.method == 'GET':
        output = '<div class="form-group">'
        output += '<label for="student-select" class="form-label">Choose Student</label>'
        output += '<select name="sname" id="student-select" class="form-control">'
        
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += f'<option value="{row[0]}">{row[0]}</option>'
        
        output += '</select>'
        output += '</div>'
        
        context = {'data1': output}
        return render(request, 'AddMarks.html', context)

def UploadMaterialAction(request):
    if request.method == 'POST':
        global uname
        material = request.POST.get('t1', False)
        desc = request.POST.get('t2', False)
        filename = request.FILES['t3'].name
        myfile = request.FILES['t3'].read()
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()        
        status = "Error in uploading material details"
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO uploadmaterial VALUES('"+uname+"','"+material+"','"+desc+"','"+filename+"','"+current_datetime+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        if db_cursor.rowcount == 1:
            status = "Material details added to database"
            if os.path.exists("StudentApp/static/files/"+filename):
                os.remove("StudentApp/static/files/"+filename)
            with open("StudentApp/static/files/"+filename, "wb") as file:
                file.write(myfile)
            file.close()             
        context= {'data': status}
        return render(request, 'UploadMaterial.html', context)

def UploadMaterial(request):
    if request.method == 'GET':
       return render(request, 'UploadMaterial.html', {})

def StudentScreen(request):
    if request.method == 'GET':
       return render(request, 'StudentScreen.html', {}) 
    

def CreateAssignmentsAction(request):
    if request.method == 'POST':
        global uname
        course = request.POST.get('t1', False)
        subject = request.POST.get('t2', False)
        year = request.POST.get('t3', False)
        assignment = request.POST.get('t4', False)
        desc = request.POST.get('t5', False)
        dd = request.POST.get('t6', False)
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO assignments VALUES('"+uname+"','"+course+"','"+subject+"','"+year+"','"+assignment+"','"+desc+"','"+dd+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Assignment task details successfully submitted"
        context= {'data': status}
        return render(request, 'CreateAssignments.html', context)

def CreateAssignments(request):
    if request.method == 'GET':
       return render(request, 'CreateAssignments.html', {}) 

def AddAttendanceAction(request):
    if request.method == 'POST':
        global uname
        students = request.POST.getlist('t1')
        now = datetime.now()
        current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        current_datetime = current_datetime.split(" ")
        current_datetime = current_datetime[0].strip()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        db_cursor = db_connection.cursor()
        for i in range(len(students)):
            student_sql_query = "INSERT INTO student_attendance VALUES('"+students[i]+"','"+uname+"','"+current_datetime+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
        output = "Selected Students Attendance Marked Successfully"
        context= {'data': output}
        return render(request, 'FacultyScreen.html', context)     
            

def AddAttendance(request):
    if request.method == 'GET':
        output = '<div class="form-group">'
        output += '<label class="form-label">Select Students</label>'
        output += '<select name="t1" multiple class="form-control student-select" style="height: auto; min-height: 150px; padding: 0.5rem;">'
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += f'<option value="{row[0]}">{row[0]}</option>'
        output += '</select>'
        output += '<small class="form-text text-muted mt-2">Hold Ctrl (or Cmd on Mac) to select multiple students</small>'
        output += '</div>'
        context = {'data1': output}
        return render(request, 'AddAttendance.html', context) 

def SchoolPerformance(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Teacher Name</th><th><font size="" color="black">Average Students Performance Grade</th>'
        output+='</tr>'
        scores = []
        labels = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select faculty_name, avg(subject_marks) from marks group by faculty_name")
            rows = cur.fetchall()
            for row in rows:
                scores.append(row[1])
                labels.append(row[0])
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td></tr>'
        output+= "</table></br>"        
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        plt.pie(scores, labels=labels, autopct='%1.1f%%')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.title("Teacher's Effectiveness Based on Student Performance Grades")
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)         

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})    

def StudentLogin(request):
    if request.method == 'GET':
       return render(request, 'StudentLogin.html', {})

def AddFaculty(request):
    if request.method == 'GET':
       return render(request, 'AddFaculty.html', {})

def AddStudent(request):
    if request.method == 'GET':
       return render(request, 'AddStudent.html', {})    

def index(request):
    if request.method == 'GET':
        global graph, accuracy, precision, recall, fscore
        output = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output += '</tr>'
        algorithms = ['Naive Bayes', 'XGBoost']
        for i in range(len(accuracy)):
            output += '<tr><td><font size="" color="black">' + algorithms[i] + '</td><td><font size="" color="black">' + str(accuracy[i]) + '</td>'
            output += '<td><font size="" color="black">' + str(precision[i]) + '</td><td><font size="" color="black">' + str(recall[i]) + '</td>'
            output += '<td><font size="" color="black">' + str(fscore[i]) + '</td></tr>'
        output += "</table></br>"

        # Generate the graph
        metrics_graph = generateMetricsGraph()

        context = {'data': output, 'graph': metrics_graph}
        return render(request, 'index.html', context)

def FacultyLogin(request):
    if request.method == 'GET':
       return render(request, 'FacultyLogin.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'AdminLogin.html', context)

def FacultyLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM faculty where username='"+username+"' and password='"+password+"'")
            rows = cur.fetchall()
            for row in rows:
                uname = username
                index = 1
                break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'FacultyScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'FacultyLogin.html', context)        
    
def StudentLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM student")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'StudentScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'StudentLogin.html', context)

def ViewFaculty(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Gender</th>'
        output+='<th><font size="" color="black">Contact No</th><th><font size="" color="black">Email ID</th>'
        output+='<th><font size="" color="black">Qualification</th><th><font size="" color="black">Experience</th>'
        output+='<th><font size="" color="black">Teaching Subjects</th>'
        output+='<th><font size="" color="black">Username</th><th><font size="" color="black">Password</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from faculty")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td><td><font size="" color="black">'+row[5]+'</td>'
                output+='<td><font size="" color="black">'+row[6]+'</td>'
                output+='<td><font size="" color="black">'+row[7]+'</td>'
                output+='<td><font size="" color="black">'+row[8]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)    

def AddFacultyAction(request):
    if request.method == 'POST':
        faculty = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        qualification = request.POST.get('t5', False)
        experience = request.POST.get('t6', False)
        teaching = request.POST.get('t7', False)
        username = request.POST.get('t8', False)
        password = request.POST.get('t9', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM faculty")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO faculty(faculty_name,gender,contact_no,email,qualification,experience,teaching_subjects,username,password) VALUES('"+faculty+"','"+gender+"','"+contact+"','"+email+"','"+qualification+"','"+experience+"','"+teaching+"','"+username+"','"+password+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Faculty details added"
        context= {'data': status}
        return render(request, 'AddFaculty.html', context)

def ViewStudent(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Student Name</th><th><font size="" color="black">Gender</th>'
        output+='<th><font size="" color="black">Contact No</th><th><font size="" color="black">Email ID</th>'
        output+='<th><font size="" color="black">Course Name</th><th><font size="" color="black">Year</th>'
        output+='<th><font size="" color="black">Username</th><th><font size="" color="black">Password</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from student")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td><td><font size="" color="black">'+row[5]+'</td>'
                output+='<td><font size="" color="black">'+row[6]+'</td>'
                output+='<td><font size="" color="black">'+row[7]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)  

def AddStudentAction(request):
    if request.method == 'POST':
        student = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        course = request.POST.get('t5', False)
        year = request.POST.get('t6', False)
        username = request.POST.get('t7', False)
        password = request.POST.get('t8', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM student")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO student(student_name,gender,contact_no,email,course,course_year,username,password) VALUES('"+student+"','"+gender+"','"+contact+"','"+email+"','"+course+"','"+year+"','"+username+"','"+password+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Student details added"
        context= {'data': status}
        return render(request, 'AddStudent.html', context)

