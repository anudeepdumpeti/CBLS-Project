# CBLS-Project 
Empowering - Student Through Choice based learning system using Machine Learning 

The project is based on choice based learning system which meanes students can choose faculty based on their experience, teaching performance and feedback given by students moreover students and faculties can communicate eachother if they get any doubts. Admin can manage students, faculties and view the performace of faculties. Faculty can add attendance, add marks, give assignments, view progress of students, view student messages, send messages to students and upload materials. Student can choose faculty, view marks, view materials, give feedback, send messages to faculty and view messages of faculties.


# **Database to create tables in SQL**

create database studentportal; # Database name can change as what you can want

use studentportal; # use database name

create table faculty(faculty_name varchar(40), gender varchar(20), contact_no varchar(20), email varchar(50), qualification varchar(65), experience varchar(45), teaching_subjects varchar(55),
username varchar(50), password varchar(50));

create table student(student_name varchar(40), gender varchar(20), contact_no varchar(20), email varchar(50), course varchar(50), course_year varchar(50), username varchar(50), password varchar(50));

create table student_attendance(student_name varchar(50), faculty_name varchar(30), attended_date varchar(30));

create table uploadmaterial(faculty_name varchar(40), material_name varchar(40), description varchar(40), filename varchar(40), upload_date varchar(40));

create table messages(sender_name varchar(50), receiver_name varchar(50), subject varchar(250), message varchar(500), message_date varchar(30));

create table marks(student_name varchar(50), faculty_name varchar(50), course_name varchar(50), course_year varchar(10), subject_name varchar(50), subject_marks double, feedback varchar(150),
upload_date varchar(40));

create table assignments(faculty_name varchar(50), course_name varchar(50), subject_name varchar(50), course_year varchar(20), assignment_task varchar(150), description varchar(250),
assignment_date varchar(30));

create table feedback(student_name varchar(40), faculty_name varchar(50), feedback varchar(150), ratings varchar(10), feedback_date varchar(20));

create table choosefaculty(faculty_name varchar(50),student_name varchar(40),class_name varchar(40));


# **Requerments**
pip install numpy, pandas, seaborn, matplotlib, scikit-learn, xgboost, django, pymysql

# **To exicute the project at first you have to runserver, open cmd on project folder then type**
python manage.py runserver then click the enter button, later copy the server link and paste it in your browser ex: http://127.0.0.1:8000/


