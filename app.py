import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
course_names = [
    "Agric Extension & Communication Technology", "Agricultural Engineering", "Agriculture Resources Economics",
    "Animal Production & Health Services", "Applied Biology", "Applied Geophysics", "Architecture",
    "Biochemistry", "Biology", "Biomedical Technology", "Biotechnology", "Building", "Civil Engineering",
    "Computer Engineering", "Computer Science", "Crop Soil & Pest Management", "Cyber Security",
    "Ecotourism & Wildlife Management", "Electrical/Electronics Engineering", "Estate Management",
    "Fisheries & Aquaculture", "Food Science & Technology", "Forestry & Wood Technology", "Human Anatomy",
    "Industrial & Production Engineering", "Industrial Chemistry", "Industrial Design", "Industrial Mathematics",
    "Information & Communication Technology", "Information Systems", "Information Technology",
    "Marine Science & Technology", "Mathematics", "Mechanical Engineering", "Medical Laboratory Science",
    "Metallurgical & Materials Engineering", "Meteorology", "Microbiology", "Mining Engineering", "Physics",
    "Physiology", "Quantity Surveying", "Integrates & Geoscience Information System", "Software Engineering",
    "Statistics", "Surveying & Geoinformatics", "Urban & Regional Planning"
]

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
    "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara"
]

learning_styles = ["Analytical Thinker", "Visual Learner", "Practical Learner", "Conceptual Learner", "Social Learner"]

grade_map = {"A1": 6, "B2": 5, "B3": 4, "C4": 3, "C5": 2, "C6": 1}

# Available O'Level subjects
olevel_subjects_options = [
    "English Language", "Mathematics", "Physics", "Chemistry", "Biology", "Economics", "Geography",
    "Agricultural Science", "Technical Drawing", "Further Mathematics", "Government", "Literature in English",
    "Fine Art", "Introduction to Agricultural Science", "Materials and Workshop Process and Machining",
    "Tractor Layout Power Unit Under Carriage and Auto Electricity", "Statistics", "Basic Electricity",
    "Spinning", "Weaving", "Surface Design and Printing", "Introduction to Building Construction",
    "Bricklaying/Block laying", "Concreting", "Wall, Floors and Ceiling Finishing", "Joinery", "Carpentry",
    "Decorative Painting", "Lining, Sign and Design", "Wall Hanging", "Colour Mixing/Matching and Glazing",
    "Ceramics", "Graphics Design", "Graphic Printing", "Basic Catering and Food Services",
    "Bakery and Confectionaries", "Hotel & Catering Crafty course (Cookery)",
    "Hotel & Catering Craft Course (Food/Drinks Services)"
]

# Sample CSV data for downloading as a template
sample_csv_data = """student_id,name,utme_score,preferred_course,utme_subjects,interests,learning_style,state_of_origin,gender,english_language_grade,mathematics_grade,physics_grade,chemistry_grade,biology_grade,economics_grade,geography_grade,agricultural_science_grade,technical_drawing_grade
001,John Adebayo,250,Biochemistry,English Language,Physics,Chemistry,Biology,Health Sciences,Analytical Thinker,Lagos,Male,A1,B3,C4,C4,C5,,,,
002,Maryam Musa,190,Agricultural Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Kano,Female,B2,B3,C4,C6,,,,,
003,Chinedu Okeke,220,Computer Science,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Anambra,Male,A1,A1,B3,C4,,,,,
004,Fatima Bello,180,Applied Biology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Kaduna,Female,B3,C4,C6,C5,B2,,,,
005,Tunde Olatunji,270,Mechanical Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Ogun,Male,A1,B2,B3,C4,,,,,
006,Aisha Ibrahim,200,Architecture,English Language,Mathematics,Physics,Fine Art,Environmental Sciences,Visual Learner,Abia,Female,B2,B3,C4,,C6,,C5,,
007,Emeka Nwosu,230,Civil Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Imo,Male,A1,A1,B3,C5,,,,,
008,Halima Sani,195,Fisheries & Aquaculture,English Language,Biology,Chemistry,Agricultural Science,Agriculture & Food Security,Conceptual Learner,Bauchi,Female,B3,C4,C6,,B2,,,,
009,Segun Adeyemi,240,Computer Engineering,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Ondo,Male,A1,B2,B3,C4,,,,,
010,Chioma Eze,210,Food Science & Technology,English Language,Chemistry,Biology,Mathematics,Agriculture & Food Security,Conceptual Learner,Enugu,Female,B2,C4,C5,B3,,,,,
011,Abdul Rahman,260,Software Engineering,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Katsina,Male,A1,A1,B2,C4,,,,,
012,Esther Ojo,185,Urban & Regional Planning,English Language,Mathematics,Geography,Economics,Environmental Sciences,Visual Learner,Ekiti,Female,B3,C5,,C6,,B2,C4,,
013,Ibrahim Yusuf,200,Industrial Mathematics,English Language,Mathematics,Physics,Economics,Technology & Innovation,Analytical Thinker,Jigawa,Male,A1,B3,C4,,,B2,,,
014,Ngozi Okonkwo,230,Microbiology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Anambra,Female,B2,C4,C5,B3,C6,,,,
015,Bolaji Ade,245,Electrical/Electronics Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Oyo,Male,A1,B2,B3,C4,,,,,
016,Zainab Lawal,190,Estate Management,English Language,Mathematics,Economics,Geography,Business & Management,Social Learner,Sokoto,Female,B3,C5,,C6,,B2,C4,,
017,Chukwudi Obi,210,Applied Geophysics,English Language,Physics,Chemistry,Mathematics,Environmental Sciences,Analytical Thinker,Delta,Male,A1,B3,C4,C5,,,,,
018,Aminat Sule,205,Biotechnology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Kwara,Female,B2,C4,C6,B3,C5,,,,
019,Femi Alabi,255,Mining Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Osun,Male,A1,B2,B3,C4,,,,,
020,Grace Adewale,195,Information Technology,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Lagos,Female,B3,C4,B2,C5,,,,,
021,Musa Danjuma,240,Physics,English Language,Physics,Mathematics,Chemistry,Technology & Innovation,Analytical Thinker,Borno,Male,A1,B3,C4,C5,,,,,
022,Amaka Igwe,200,Forestry & Wood Technology,English Language,Chemistry,Biology,Agricultural Science,Agriculture & Food Security,Conceptual Learner,Imo,Female,B2,C4,C5,,B3,,,,
023,Sani Mohammed,230,Cyber Security,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Kano,Male,A1,B2,B3,C4,,,,,
024,Funmi Adegoke,190,Human Anatomy,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Ogun,Female,B3,C5,C6,B2,C4,,,,
025,Victor Eke,250,Industrial & Production Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Abia,Male,A1,B2,B3,C4,,,,,
026,Rukayat Ahmed,210,Statistics,English Language,Mathematics,Physics,Economics,Technology & Innovation,Analytical Thinker,Kaduna,Female,B2,C4,B3,,,C5,,,
027,Chike Nwankwo,245,Metallurgical & Materials Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Enugu,Male,A1,B2,B3,C4,,,,,
028,Hauwa Umar,195,Ecotourism & Wildlife Management,English Language,Biology,Chemistry,Geography,Agriculture & Food Security,Conceptual Learner,Bauchi,Female,B3,C5,,C6,,B2,C4,,
029,Taiwo Ogunleye,230,Information & Communication Technology,English Language,Mathematics,Economics,Government,Business & Management,Social Learner,Oyo,Male,B2,C4,,,B3,,C5,
030,Prisca Okoro,200,Medical Laboratory Science,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Delta,Female,B3,C4,C5,B2,C6,,,,
031,Usman Bello,260,Building,English Language,Mathematics,Physics,Chemistry,Environmental Sciences,Practical Learner,Jigawa,Male,A1,B2,B3,C4,,,,,
032,Chiamaka Udeh,190,Industrial Chemistry,English Language,Chemistry,Mathematics,Biology,Health Sciences,Conceptual Learner,Anambra,Female,B2,C5,C6,B3,C4,,,,
033,Lekan Adeyemo,240,Surveying & Geoinformatics,English Language,Mathematics,Physics,Geography,Environmental Sciences,Visual Learner,Ekiti,Male,A1,B3,C4,,C5,,B2,,
034,Aminu Sani,210,Meteorology,English Language,Mathematics,Physics,Geography,Environmental Sciences,Analytical Thinker,Sokoto,Male,B2,C4,B3,,C6,,C5,,
035,Ebuka Nnamdi,255,Computer Engineering,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Practical Learner,Imo,Male,A1,B2,B3,C4,,,,,
036,Fatima Abubakar,200,Physiology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Kano,Female,B3,C4,C5,B2,C6,,,,
037,Kayode Olanrewaju,230,Industrial Design,English Language,Mathematics,Chemistry,Fine Art,Engineering & Design,Visual Learner,Ogun,Male,B2,C4,B3,,C5,,C6,,
038,Chinelo Okoye,195,Agric Extension & Communication Technology,English Language,Chemistry,Biology,Agricultural Science,Agriculture & Food Security,Conceptual Learner,Enugu,Female,B3,C5,,C6,B2,,,,
039,Suleiman Idris,245,Mechanical Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Kaduna,Male,A1,B2,B3,C4,,,,,
040,Funke Adesina,210,Information Systems,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Lagos,Female,B2,C4,B3,C5,,,,,
041,Obinna Eze,230,Civil Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Abia,Male,A1,B2,B3,C4,,,,,
042,Halima Musa,190,Biomedical Technology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Borno,Female,B3,C5,C6,B2,C4,,,,
043,Tolu Adekunle,250,Computer Science,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Oyo,Male,A1,B2,B3,C4,,,,,
044,Nkechi Obi,205,Food Science & Technology,English Language,Chemistry,Biology,Mathematics,Agriculture & Food Security,Conceptual Learner,Anambra,Female,B2,C4,C5,B3,,,,,
045,Yusuf Adamu,240,Electrical/Electronics Engineering,English Language,Mathematics,Physics,Chemistry,Engineering & Design,Practical Learner,Kano,Male,A1,B2,B3,C4,,,,,
046,Chioma Nwosu,195,Urban & Regional Planning,English Language,Mathematics,Geography,Economics,Environmental Sciences,Visual Learner,Ekiti,Female,B3,C5,,C6,,B2,C4,,
047,Adebayo Tunde,230,Applied Biology,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Ogun,Male,B2,C4,C5,B3,C6,,,,
048,Zainab Yusuf,210,Biochemistry,English Language,Biology,Chemistry,Physics,Health Sciences,Conceptual Learner,Kaduna,Female,B3,C4,C5,B2,C6,,,,
049,Chukwuma Igwe,245,Software Engineering,English Language,Mathematics,Physics,Chemistry,Technology & Innovation,Analytical Thinker,Imo,Male,A1,B2,B3,C4,,,,,
050,Rukayat Bello,200,Architecture,English Language,Mathematics,Physics,Fine Art,Environmental Sciences,Visual Learner,Lagos,Female,B2,B3,C4,,C6,,C5,,
"""

# Initialize session state
if 'form_key_counter' not in st.session_state:
    st.session_state.form_key_counter = 0
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'eligible_courses_df' not in st.session_state:
    st.session_state.eligible_courses_df = pd.DataFrame()
if 'invalid_rows' not in st.session_state:
    st.session_state.invalid_rows = []

def load_jamb_data():
    return pd.DataFrame({
        'course': course_names,
        'faculty': (
            ['Agriculture'] * 9 +  # Agric Extension, Agricultural Engineering, Agriculture Resources Economics,
                                  # Animal Production, Crop Soil & Pest Management, Ecotourism & Wildlife Management,
                                  # Fisheries & Aquaculture, Food Science & Technology, Forestry & Wood Technology
            ['Science'] * 17 +    # Applied Biology, Applied Geophysics, Biochemistry, Biology, Biomedical Technology,
                                  # Biotechnology, Human Anatomy, Industrial Chemistry, Industrial Mathematics,
                                  # Marine Science & Technology, Mathematics, Medical Laboratory Science, Meteorology,
                                  # Microbiology, Physics, Physiology, Statistics
            ['Environmental Sciences'] * 7 +  # Architecture, Building, Estate Management, Quantity Surveying,
                                              # Integrates & Geoscience Information System, Surveying & Geoinformatics,
                                              # Urban & Regional Planning
            ['Engineering'] * 8 +  # Civil Engineering, Computer Engineering, Computer Science,
                                   # Electrical/Electronics Engineering, Industrial & Production Engineering,
                                   # Mechanical Engineering, Metallurgical & Materials Engineering, Mining Engineering
            ['Technology'] * 6    # Cyber Security, Industrial Design, Information & Communication Technology,
                                  # Information Systems, Information Technology, Software Engineering
        ),
        'total_2017': np.random.randint(100, 1000, size=len(course_names)),
        'total_2018': np.random.randint(100, 1000, size=len(course_names))
    })

def load_neco_data():
    return pd.DataFrame({
        'state': NIGERIAN_STATES,
        'male_pass_rate': np.random.uniform(50, 90, len(NIGERIAN_STATES)),
        'female_pass_rate': np.random.uniform(50, 90, len(NIGERIAN_STATES))
    })

def get_course_requirements():
    requirements = {
        "Agric Extension & Communication Technology": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Mathematics": "C6", "Biology/Agricultural Science": "C6"}
        },
        "Agricultural Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Chemistry", "Physics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Agriculture Resources Economics": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Mathematics": "C6", "Biology": "C6"}
        },
        "Animal Production & Health Services": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Mathematics": "C6", "Biology/Agricultural Science": "C6"}
        },
        "Applied Biology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Chemistry/Physics/Mathematics/Biology/Geography"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Biology": "C6", "Chemistry": "C6", "Mathematics": "C6"}
        },
        "Applied Geophysics": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Physics/Chemistry/Biology/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Architecture": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Physics", "Mathematics", "Chemistry/Geography/Art/Biology/Economics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Biochemistry": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Mathematics": "C6", "Physics": "C6", "Biology": "C6"}
        },
        "Biology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Biology": "C6", "Chemistry": "C6", "Mathematics/Physics": "C6"}
        },
        "Biomedical Technology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Physics", "Chemistry", "Biology"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6", "Biology": "C6"}
        },
        "Biotechnology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Any Science"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Biology": "C6", "Chemistry": "C6", "Physics": "C6"}
        },
        "Building": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Civil Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Physics", "Chemistry", "Mathematics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Computer Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Further Mathematics": "C6", "Chemistry": "C6", "Physics": "C6"}
        },
        "Computer Science": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Biology/Chemistry/Agric Science/Economics/Geography"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Crop Soil & Pest Management": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Mathematics/Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Biology/Agricultural Science": "C6", "Mathematics": "C6"}
        },
        "Cyber Security": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Biology/Chemistry/Agric Science/Economics/Geography"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Ecotourism & Wildlife Management": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Biology/Agricultural Science", "Chemistry/Geography/Economics/Mathematics/Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Biology/Agricultural Science": "C6"}
        },
        "Electrical/Electronics Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Estate Management": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Mathematics", "Economics", "Any Subject"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6"}
        },
        "Fisheries & Aquaculture": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Mathematics/Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Biology/Agricultural Science": "C6", "Mathematics": "C6"}
        },
        "Food Science & Technology": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Mathematics/Physics", "Biology/Agricultural Science"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6", "Biology/Agricultural Science": "C6"}
        },
        "Forestry & Wood Technology": {
            "faculty": "Agriculture",
            "required_subjects": ["English Language", "Chemistry", "Biology/Agricultural Science", "Mathematics/Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Chemistry": "C6", "Biology/Agricultural Science": "C6", "Mathematics": "C6"}
        },
        "Human Anatomy": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Biology": "C6", "Chemistry": "C6", "Physics": "C6"}
        },
        "Industrial & Production Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Industrial Chemistry": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Chemistry", "Mathematics", "Physics/Biology/Agricultural Science"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6", "Physics": "C6", "Biology/Agricultural Science": "C6"}
        },
        "Industrial Design": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Chemistry", "Mathematics", "Fine Arts/Physics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6", "Fine Art": "C6"}
        },
        "Industrial Mathematics": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Mathematics", "Physics/Chemistry/Economics/Biology/Agricultural Science"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Information & Communication Technology": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Mathematics", "Economics", "Account/Commerce/Government"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Economics": "C6", "Mathematics": "C6"}
        },
        "Information Systems": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Biology/Chemistry/Agric Science/Economics/Geography"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Information Technology": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Biology/Chemistry/Agric Science/Economics/Geography"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Marine Science & Technology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Physics/Chemistry/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Biology": "C6"}
        },
        "Mathematics": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Mathematics", "Physics/Chemistry/Economics/Geography"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics/Chemistry": "C6"}
        },
        "Mechanical Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Medical Laboratory Science": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Physics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6", "Biology": "C6", "Physics": "C6"}
        },
        "Metallurgical & Materials Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Physics", "Chemistry", "Mathematics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Meteorology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Chemistry/Geography"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Microbiology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Biology", "Chemistry", "Physics/Mathematics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Chemistry": "C6", "Biology": "C6", "Physics": "C6"}
        },
        "Mining Engineering": {
            "faculty": "Engineering",
            "required_subjects": ["English Language", "Chemistry", "Mathematics", "Physics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Physics": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Physics", "Mathematics", "Chemistry/Biology"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Physics": "C6", "Chemistry": "C6", "Mathematics": "C6"}
        },
        "Physiology": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Physics", "Chemistry", "Biology"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6", "Biology": "C6"}
        },
        "Quantity Surveying": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Physics", "Mathematics", "Chemistry/Geography/Art/Biology/Economics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Integrates & Geoscience Information System": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Physics", "Mathematics", "Chemistry/Geography/Art/Biology/Economics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Software Engineering": {
            "faculty": "Technology",
            "required_subjects": ["English Language", "Mathematics", "Physics", "Biology/Chemistry/Agric Science/Economics/Geography"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Statistics": {
            "faculty": "Science",
            "required_subjects": ["English Language", "Mathematics", "Physics/Chemistry/Economics"],
            "min_utme": 180,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"}
        },
        "Surveying & Geoinformatics": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Physics", "Mathematics", "Chemistry/Geography/Art/Biology/Economics"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"}
        },
        "Urban & Regional Planning": {
            "faculty": "Environmental Sciences",
            "required_subjects": ["English Language", "Mathematics", "Geography", "Economics/Physics/Chemistry"],
            "min_utme": 200,
            "min_olevel": {"English Language": "C6", "Mathematics": "C6", "Geography": "C6"}
        }
    }
    return requirements

def generate_admission_letter(student_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Federal University of Technology, Akure", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Admission Letter for {student_data['name']}", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Student ID: {student_data['student_id']}", styles['Normal']))
    story.append(Paragraph(f"Course: {student_data['admitted_course']}", styles['Normal']))
    story.append(Paragraph(f"Admission Type: {student_data['admission_type']}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Dear {student_data['name']},", styles['Normal']))
    story.append(Paragraph("Congratulations on your admission to the Federal University of Technology, Akure!", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

def predict_placement_enhanced(utme_score, olevel_subjects, utme_subjects, interests, learning_style, state, gender, preferred_course):
    requirements = get_course_requirements()
    neco_data = load_neco_data()
    eligible_courses = []
    failure_reasons = []
    
    for course in course_names:
        req = requirements[course]
        meets_utme = utme_score >= req['min_utme']
        
        # Handle flexible subject requirements (e.g., "Physics/Mathematics" or "Any Science")
        meets_subjects = True
        for subj in req['required_subjects']:
            if '/' in subj:
                options = subj.split('/')
                if not any(opt in utme_subjects for opt in options):
                    meets_subjects = False
                    break
            elif subj in ["Any Science", "Any Subject"]:
                non_english_subjects = [s for s in utme_subjects if s != "English Language"]
                if len(non_english_subjects) < 3:  # Ensure at least 3 non-English subjects
                    meets_subjects = False
                    break
            elif subj not in utme_subjects:
                meets_subjects = False
                break
        
        # Handle O'Level requirements with alternatives (e.g., Biology/Agricultural Science)
        meets_olevel = True
        for subject, min_grade in req['min_olevel'].items():
            if '/' in subject:
                options = subject.split('/')
                if not any(opt in olevel_subjects and olevel_subjects[opt] >= grade_map[min_grade] for opt in options):
                    meets_olevel = False
                    break
            elif subject not in olevel_subjects or olevel_subjects[subject] < grade_map[min_grade]:
                meets_olevel = False
                break
        
        if not meets_utme:
            failure_reasons.append(f"{course}: UTME score {utme_score} below required {req['min_utme']}")
        if not meets_subjects:
            missing_subjects = [subj for subj in req['required_subjects'] if '/' not in subj and subj not in utme_subjects and subj not in ["Any Science", "Any Subject"]]
            failure_reasons.append(f"{course}: Missing UTME subjects: {', '.join(missing_subjects)}")
        if not meets_olevel:
            missing_olevel = [subject for subject, grade in req['min_olevel'].items() if subject not in olevel_subjects or olevel_subjects[subject] < grade_map[grade]]
            failure_reasons.append(f"{course}: O'Level requirements not met for: {', '.join(missing_olevel)}")
        
        if meets_utme and meets_subjects and meets_olevel:
            # Interest score: Higher if course matches any interest (make it more granular/realistic)
            interest_score = sum(0.1 for interest in interests if interest.lower() in course.lower()) + 0.1  # Base 0.1, +0.1 per match, up to 0.4 if multiple
            interest_score = min(interest_score, 0.4)  # Cap for balance
            
            # Learning style: Expand for realism (e.g., Visual for Design, Analytical for Math/Science)
            learning_style_score = 0.1
            if learning_style == 'Analytical Thinker' and any(word in course for word in ['Mathematics', 'Physics', 'Statistics', 'Geophysics']):
                learning_style_score = 0.3
            elif learning_style == 'Practical Learner' and 'Engineering' in course:
                learning_style_score = 0.3
            elif learning_style == 'Visual Learner' and any(word in course for word in ['Architecture', 'Design', 'Urban']):
                learning_style_score = 0.3
            elif learning_style == 'Conceptual Learner' and any(word in course for word in ['Science', 'Biology', 'Chemistry']):
                learning_style_score = 0.3
            elif learning_style == 'Social Learner' and any(word in course for word in ['Communication', 'Management', 'Economics']):
                learning_style_score = 0.3
            
            # Diversity: Real-life catchment area bonus (first 5 states as example)
            diversity_score = 0.1 if state in NIGERIAN_STATES[:5] else 0.05
            
            # O-level pass rate (normalized 0-1)
            olevel_pass_rate = np.mean([olevel_subjects[subj] for subj in olevel_subjects]) / 6
            
            eligible_courses.append({
                'course': course,
                'interest_score': interest_score,
                'learning_style_score': learning_style_score,
                'diversity_score': diversity_score,
                'olevel_pass_rate': olevel_pass_rate
            })

    if not eligible_courses:
        return {
            'predicted_program': 'UNASSIGNED',
            'reason': 'No eligible courses found based on your scores and subjects.',
            'suggestions': ['Improve UTME score', 'Ensure required O-level and UTME subjects are met'],
            'all_eligible': pd.DataFrame(),
            'score': 0,
            'interest_alignment': False,
            'failure_reasons': failure_reasons
        }

    eligible_df = pd.DataFrame(eligible_courses)
    
    # Normalized UTME (0-1)
    utme_norm = utme_score / 400.0
    
    # Composite score: Weighted sum (emphasize interest and learning style for "intelligence")
    eligible_df['total_score'] = (
        0.3 * eligible_df['interest_score'] +  # 30% weight on interests
        0.25 * eligible_df['learning_style_score'] +  # 25% on learning fit
        0.15 * eligible_df['diversity_score'] +  # 15% on diversity
        0.15 * eligible_df['olevel_pass_rate'] +  # 15% on O-level
        0.15 * utme_norm  # 15% on UTME
    )
    
    # Bonus for preferred course (real-life priority)
    eligible_df.loc[eligible_df['course'] == preferred_course, 'total_score'] += 0.1
    
    # Sort by total_score descending
    eligible_df = eligible_df.sort_values(by='total_score', ascending=False)
    
    if eligible_df.empty:
        return {
            'predicted_program': 'UNASSIGNED',
            'reason': 'No eligible courses after scoring.',
            'suggestions': ['Improve scores or adjust interests/subjects'],
            'all_eligible': pd.DataFrame(),
            'score': 0,
            'interest_alignment': False,
            'failure_reasons': failure_reasons
        }

    top_course = eligible_df.iloc[0]['course']
    top_score = eligible_df.iloc[0]['total_score']
    interest_alignment = eligible_df.iloc[0]['interest_score'] > 0.1

    return {
        'predicted_program': top_course,
        'reason': f"Eligible for {top_course} based on best fit to your scores, interests, and learning style.",
        'all_eligible': eligible_df,
        'score': top_score,
        'interest_alignment': interest_alignment,
        'failure_reasons': failure_reasons
    }

# Load data and model
jamb_data = load_jamb_data()
neco_data = load_neco_data()
course_capacities = {course: max(50, int(jamb_data[jamb_data['course'] == course]['total_2017'].sum() / 10)) for course in course_names}

async def process_with_timeout(coro, timeout: float = 300):
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        # Ensure result is a tuple of (admission_results, eligible_courses_df, invalid_rows)
        if not isinstance(result, tuple) or len(result) != 3:
            logger.error("process_csv_applications returned invalid result: %s", result)
            return [], pd.DataFrame(), [{"row": 0, "student_id": "N/A", "error": "Invalid processing result"}]
        return result
    except asyncio.TimeoutError:
        logger.error("Processing timed out after %s seconds", timeout)
        st.error(f"Processing timed out after {timeout} seconds.")
        return [], pd.DataFrame(), [{"row": 0, "student_id": "N/A", "error": "Processing timed out"}]
    except Exception as e:
        logger.error("Error in process_with_timeout: %s", str(e))
        st.error(f"Error in processing: {str(e)}")
        return [], pd.DataFrame(), [{"row": 0, "student_id": "N/A", "error": str(e)}]

async def process_csv_applications(df, course_capacities, jamb_data, neco_data, update_progress):
    logger.info("Starting batch processing for %d applications", len(df))
    required_columns = [
        'student_id', 'name', 'utme_score', 'preferred_course', 'utme_subjects',
        'interests', 'learning_style', 'state_of_origin', 'gender'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {', '.join(missing_cols)}"
        logger.error(error_msg)
        return [], pd.DataFrame(), [{"row": 0, "student_id": "N/A", "error": error_msg}]

    admission_results = []
    eligible_courses_list = []
    invalid_rows = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        try:
            student_id = row.get('student_id', 'Unknown')
            name = row.get('name', '').strip()
            utme_score = row.get('utme_score', 0)
            preferred_course = row.get('preferred_course', '').strip()
            utme_subjects = row.get('utme_subjects', '').split(',')
            utme_subjects = [s.strip() for s in utme_subjects if s.strip()]
            interests = row.get('interests', '').split(',')
            interests = [i.strip() for i in interests if i.strip()]
            learning_style = row.get('learning_style', '').strip()
            state = row.get('state_of_origin', '').strip()
            gender = row.get('gender', '').strip()

            olevel_subjects = {}
            for col in df.columns:
                if '_grade' in col:
                    subject = col.replace('_grade', '').replace('_', ' ').title()
                    if pd.notna(row[col]) and row[col] in grade_map:
                        olevel_subjects[subject] = grade_map[row[col]]

            if not name:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Missing or invalid name"})
                continue
            if utme_score <= 0 or utme_score > 400:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid UTME score"})
                continue
            if len(utme_subjects) != 4 or "English Language" not in utme_subjects:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Must have exactly 4 UTME subjects including English Language"})
                continue
            if len(olevel_subjects) < 5:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Must have at least 5 O'Level subjects"})
                continue
            if preferred_course not in course_names:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid preferred course"})
                continue
            if state not in NIGERIAN_STATES:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid state of origin"})
                continue
            if gender not in ["Male", "Female"]:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid gender"})
                continue
            if learning_style not in learning_styles:
                invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid learning style"})
                continue

            prediction = predict_placement_enhanced(
                utme_score,
                olevel_subjects,
                utme_subjects,
                interests,
                learning_style,
                state,
                gender,
                preferred_course
            )

            status = "Admitted" if prediction['predicted_program'] != "UNASSIGNED" else "Not Admitted"
            admission_type = "Merit" if prediction['predicted_program'] == preferred_course else "Alternative" if status == "Admitted" else "N/A"
            reason = prediction['reason']
            alternatives = ", ".join(prediction['all_eligible']['course'].tolist()) if not prediction['all_eligible'].empty else "None"
            suggested_alternatives = ", ".join(prediction['all_eligible']['course'].head(3).tolist()) if not prediction['all_eligible'].empty else "None"

            admission_results.append({
                "student_id": student_id,
                "name": name,
                "admitted_course": prediction['predicted_program'],
                "status": status,
                "admission_type": admission_type,
                "score": prediction['score'],
                "rank": prediction['all_eligible'].index[prediction['all_eligible']['course'] == prediction['predicted_program']].tolist()[0] + 1 if status == "Admitted" else 0,
                "reason": reason,
                "original_preference": preferred_course,
                "recommendation_reason": "Best match based on rule-based scoring" if status == "Admitted" else "N/A",
                "available_alternatives": alternatives,
                "suggested_alternatives": suggested_alternatives
            })

            if not prediction['all_eligible'].empty:
                for _, eligible_row in prediction['all_eligible'].iterrows():
                    eligible_courses_list.append({
                        "student_id": student_id,
                        "name": name,
                        "course": eligible_row['course'],
                        "score": eligible_row['total_score'],
                        "interest_score": eligible_row['interest_score'],
                        "learning_style_score": eligible_row['learning_style_score'],
                        "diversity_score": eligible_row['diversity_score'],
                        "olevel_pass_rate": eligible_row['olevel_pass_rate']
                    })

            await update_progress((idx + 1) / total_rows)

        except Exception as e:
            logger.error("Error processing row %d: %s", idx + 2, str(e))
            invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": str(e)})

    eligible_courses_df = pd.DataFrame(eligible_courses_list)
    return admission_results, eligible_courses_df, invalid_rows

def create_comprehensive_admission_report(admission_results, input_df, course_capacities):
    detailed_results = pd.DataFrame(admission_results)
    total_applicants = len(input_df)
    admitted = len([r for r in admission_results if r['status'] == "Admitted"])
    merit_admissions = len([r for r in admission_results if r['admission_type'] == "Merit"])
    
    summary_stats = {
        "Admission Rate": (admitted / total_applicants * 100) if total_applicants > 0 else 0,
        "Merit Admission Rate": (merit_admissions / admitted * 100) if admitted > 0 else 0,
        "Alternative Admission Rate": ((admitted - merit_admissions) / admitted * 100) if admitted > 0 else 0
    }
    
    course_breakdown = []
    for course in course_names:
        admitted_count = len([r for r in admission_results if r['admitted_course'] == course])
        scores = [r['score'] for r in admission_results if r['admitted_course'] == course]
        course_breakdown.append({
            "admitted_course": course,
            "Students_Admitted": admitted_count,
            "Avg_Score": np.mean(scores) if scores else 0,
            "Min_Score": np.min(scores) if scores else 0,
            "Max_Score": np.max(scores) if scores else 0,
            "Capacity": course_capacities.get(course, 50),
            "Utilization_Rate": (admitted_count / course_capacities.get(course, 50) * 100) if course_capacities.get(course, 50) > 0 else 0
        })
    
    course_breakdown = pd.DataFrame(course_breakdown)
    return detailed_results, summary_stats, course_breakdown

def analyze_student_demand(input_df, jamb_data, neco_data):
    course_demand = {}
    for course in course_names:
        faculty = get_course_requirements()[course]['faculty']
        demand_score = jamb_data[jamb_data['faculty'] == faculty]['total_2017'].sum() / jamb_data['total_2017'].sum() if not jamb_data['total_2017'].empty else 0
        course_demand[course] = {
            'demand_score': demand_score,
            'applicants': len([r for r in input_df.get('preferred_course', []) if r == course])
        }
    return course_demand

def optimize_course_capacities(demand, current_capacities):
    optimization = {}
    total_demand = sum(d['demand_score'] for d in demand.values()) or 1
    total_capacity = sum(current_capacities.values()) or 1
    
    for course, data in demand.items():
        current = current_capacities.get(course, 50)
        suggested = max(50, int((data['demand_score'] / total_demand) * total_capacity * 1.2))
        priority = 'High' if data['applicants'] > current else 'Low'
        optimization[course] = {
            'current_capacity': current,
            'suggested_capacity': suggested,
            'priority': priority
        }
    return optimization

def calculate_capacity_utilization(admission_results, course_capacities):
    utilization = {}
    for course in course_names:
        admitted = len([r for r in admission_results if r['admitted_course'] == course])
        capacity = course_capacities.get(course, 50)
        rate = (admitted / capacity * 100) if capacity > 0 else 0
        status = 'Optimal' if rate <= 100 else 'Overcapacity'
        utilization[course] = {
            'Utilization Rate': rate,
            'Status': status
        }
    return utilization

async def main():
    st.set_page_config(page_title="FUTA Intelligent Admission Management System", layout="wide")
    st.title("FUTA Intelligent Admission Management System")
    st.markdown("Welcome to the FUTA Intelligent Admission Management System. Use the tabs below to predict admissions, process batch applications, view analytics, or access help.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Individual Admission Prediction",
        "Batch Admission Processing",
        "Analytics & Insights",
        "Help & FAQ"
    ])

    with tab1:
        st.header("Individual Admission Prediction")
        st.markdown("Enter your details to predict your admission eligibility and course placement.")
        st.warning("Select at least 5 O'Level subjects with grades A1 to C6. UTME subjects must include English Language.")
        
        with st.form(key=f"admission_form_{st.session_state.form_key_counter}"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name")
                utme_score = st.number_input("UTME Score (0-400)", min_value=0, max_value=400, step=1)
                preferred_course = st.selectbox("Preferred Course", options=course_names)
                state = st.selectbox("State of Origin", options=NIGERIAN_STATES)
                gender = st.selectbox("Gender", options=["Male", "Female"])
            
            with col2:
                utme_subjects = st.multiselect(
                    "UTME Subjects (Select exactly 4, including English Language)",
                    options=["English Language", "Mathematics", "Physics", "Chemistry", "Biology", "Economics", "Geography",
                             "Agricultural Science", "Fine Art", "Account", "Commerce", "Government"],
                    default=["English Language"]
                )
                interests = st.multiselect(
                    "Interests (Select up to 3)",
                    options=["Technology & Innovation", "Engineering & Design", "Environmental Sciences", 
                             "Business & Management", "Health Sciences", "Agriculture & Food Security"],
                    max_selections=3
                )
                learning_style = st.selectbox("Learning Style", options=learning_styles)
            
            st.subheader("O'Level Subjects and Grades")
            st.markdown("Select up to 9 O'Level subjects and their grades.")
            olevel_subjects = {}
            selected_subjects = st.multiselect(
                "Select O'Level Subjects (at least 5)",
                options=olevel_subjects_options,
                key=f"olevel_subjects_{st.session_state.form_key_counter}"
            )
            
            col_olevel1, col_olevel2 = st.columns(2)
            for i, subject in enumerate(selected_subjects):
                col = col_olevel1 if i % 2 == 0 else col_olevel2
                with col:
                    grade = st.selectbox(
                        f"{subject} Grade",
                        options=["", "A1", "B2", "B3", "C4", "C5", "C6"],
                        key=f"{subject}_grade_{st.session_state.form_key_counter}"
                    )
                    if grade:
                        olevel_subjects[subject] = grade_map[grade]
            
            col_submit, col_reset = st.columns(2)
            with col_submit:
                submit_button = st.form_submit_button("Predict Admission")
            with col_reset:
                reset_button = st.form_submit_button("Reset Form")

            if reset_button:
                st.session_state.form_key_counter += 1
                st.rerun()

            if submit_button:
                logger.info("Individual admission form submitted: name=%s, utme_score=%s, utme_subjects=%s, interests=%s, olevel_subjects=%s",
                            name, utme_score, utme_subjects, interests, olevel_subjects)
                if not name or name.strip() == "":
                    st.error("Please enter a valid full name.")
                elif utme_score <= 0:
                    st.error("Please enter a valid UTME score greater than 0.")
                elif not utme_subjects or len(utme_subjects) != 4:
                    st.error("Please select exactly 4 UTME subjects, including English Language.")
                elif "English Language" not in utme_subjects:
                    st.error("English Language is a mandatory UTME subject.")
                elif not olevel_subjects or len(olevel_subjects) < 5:
                    st.error("Please select at least 5 O'Level subjects with valid grades.")
                else:
                    try:
                        prediction = predict_placement_enhanced(
                            utme_score,
                            olevel_subjects,
                            utme_subjects,
                            interests,
                            learning_style,
                            state,
                            gender,
                            preferred_course
                        )
                        if prediction['predicted_program'] == "UNASSIGNED":
                            st.error(prediction['reason'])
                            if 'suggestions' in prediction:
                                st.subheader("Suggestions for Improvement")
                                for suggestion in prediction['suggestions']:
                                    st.markdown(f"- {suggestion}")
                            if 'failure_reasons' in prediction and prediction['failure_reasons']:
                                st.subheader("Reasons for Ineligibility")
                                for reason in prediction['failure_reasons'][:5]:  # Limit to top 5 for brevity
                                    st.markdown(f"- {reason}")
                        else:
                            st.success(f"Congratulations! You are predicted to be admitted to **{prediction['predicted_program']}**")
                            st.write(f"**Prediction Score**: {prediction['score']:.2f}")
                            st.write(f"**Reason**: {prediction['reason']}")
                            if prediction['interest_alignment']:
                                st.write("✅ This course aligns with your interests!")
                            if not prediction['all_eligible'].empty:
                                st.subheader("Other Eligible Courses")
                                st.dataframe(
                                    prediction['all_eligible'][['course', 'total_score', 'interest_score', 'learning_style_score']],
                                    use_container_width=True,
                                    column_config={
                                        "course": "Course",
                                        "total_score": st.column_config.NumberColumn("Score", format="%.2f"),
                                        "interest_score": st.column_config.NumberColumn("Interest Score", format="%.2f"),
                                        "learning_style_score": st.column_config.NumberColumn("Learning Style Score", format="%.2f")
                                    }
                                )
                                student_data = {
                                    'name': name,
                                    'student_id': f"FUTA/{datetime.now().year}/{np.random.randint(1000, 9999)}",
                                    'admitted_course': prediction['predicted_program'],
                                    'admission_type': 'Merit' if prediction['predicted_program'] == preferred_course else 'Alternative'
                                }
                                pdf_buffer = generate_admission_letter(student_data)
                                st.download_button(
                                    label="Download Admission Letter",
                                    data=pdf_buffer,
                                    file_name=f"admission_letter_{name.replace(' ', '_')}.pdf",
                                    mime="application/pdf"
                                )
                    except Exception as e:
                        logger.error("Error in prediction: %s", str(e))
                        st.error("An error occurred during prediction. Please try again.")
                        st.write(f"Error details: {str(e)}")

        st.write("---")
        st.info("Use the 'Reset Form' button to clear inputs and start a new prediction.")

    with tab2:
        st.header("Batch Admission Processing")
        st.markdown("Upload a CSV file to process multiple admission applications at once, or download a sample CSV to test the system.")

    # Download button for sample CSV (outside any form)
    st.download_button(
        label="Download Sample CSV (50 Students)",
        data=sample_csv_data,
        file_name="sample_students.csv",
        mime="text/csv",
        key="sample_csv_download"
    )

    # File uploader for batch processing
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key=f"batch_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file:
        try:
            # Read CSV with robust parsing and converter for utme_score
            df = pd.read_csv(
                uploaded_file,
                skipinitialspace=True,
                quoting=csv.QUOTE_ALL,
                converters={'utme_score': lambda x: pd.to_numeric(x, errors='coerce')}
            )
            if df.empty:
                st.error("Uploaded CSV is empty. Please upload a valid CSV file.")
                st.session_state.uploader_key += 1
                st.rerun()

            # Log the first few rows for debugging
            logger.info("CSV head:\n%s", df.head().to_string())

            # Define required and optional columns
            required_columns = [
                'student_id', 'name', 'utme_score', 'preferred_course', 'utme_subjects',
                'interests', 'learning_style', 'state_of_origin', 'gender', 'english_language_grade'
            ]
            optional_columns = [
                'mathematics_grade', 'physics_grade', 'chemistry_grade', 'biology_grade',
                'economics_grade', 'geography_grade', 'agricultural_science_grade', 'technical_drawing_grade'
            ]

            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.session_state.uploader_key += 1
                st.rerun()

            # Add missing optional columns with default 0
            for col in optional_columns:
                if col not in df.columns:
                    df[col] = 0

            # Fill NaN in utme_score with -1 to mark invalid rows
            df['utme_score'] = df['utme_score'].fillna(-1)

            # Identify rows with non-numeric or invalid utme_score
            invalid_utme_rows = df[
                (df['utme_score'].isna()) |
                (df['utme_score'] < 0) |
                (df['utme_score'] > 400)
            ]
            if not invalid_utme_rows.empty:
                st.warning("Some rows have invalid UTME scores (non-numeric, missing, or out of range 0–400). These will be reported in the invalid rows output.")
                invalid_utme_df = invalid_utme_rows[['student_id', 'name', 'utme_score']].copy()
                invalid_utme_df['error'] = invalid_utme_df['utme_score'].apply(
                    lambda x: "Non-numeric or missing" if pd.isna(x) or x == -1 else "Out of range (0–400)"
                )
                st.subheader("Rows with Invalid UTME Scores")
                st.dataframe(
                    invalid_utme_df,
                    use_container_width=True,
                    column_config={
                        'student_id': "Student ID",
                        'name': "Name",
                        'utme_score': "UTME Score",
                        'error': "Error Message"
                    }
                )

            # Filter out invalid rows for processing
            valid_df = df[
                (~df['utme_score'].isna()) &
                (df['utme_score'] >= 0) &
                (df['utme_score'] <= 400)
            ].copy()

            if valid_df.empty:
                st.error("No valid rows to process after filtering invalid UTME scores.")
                invalid_rows = [
                    {"row": idx + 2, "student_id": row['student_id'], "error": "Invalid UTME score"}
                    for idx, row in invalid_utme_rows.iterrows()
                ]
                invalid_csv_buffer = io.StringIO()
                pd.DataFrame(invalid_rows).to_csv(invalid_csv_buffer, index=False)
                st.download_button(
                    label="Download Invalid Rows Report",
                    data=invalid_csv_buffer.getvalue(),
                    file_name="invalid_rows.csv",
                    mime="text/csv",
                    key="invalid_rows_download_initial"
                )
                st.session_state.uploader_key += 1
                st.rerun()

            # Convert utme_subjects and interests to lists
            valid_df['utme_subjects'] = valid_df['utme_subjects'].apply(
                lambda x: x.split(',') if isinstance(x, str) and x.strip() else []
            )
            valid_df['interests'] = valid_df['interests'].apply(
                lambda x: x.split(',') if isinstance(x, str) and x.strip() else []
            )

            # Convert O'Level grades to numeric scores (A1=6, B2=5, B3=4, C4=3, C5=2, C6=1)
            for col in optional_columns + ['english_language_grade']:
                if col in valid_df.columns:
                    valid_df[col] = valid_df[col].apply(lambda x: grade_map.get(str(x).strip(), 0)).astype(int)

            # Additional validation for required fields
            invalid_rows = []
            for idx, row in valid_df.iterrows():
                student_id = row.get('student_id', 'Unknown')
                if pd.isna(row['student_id']) or not str(row['student_id']).strip():
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Missing or invalid student_id"})
                elif pd.isna(row['name']) or not str(row['name']).strip():
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Missing or invalid name"})
                elif len(row['utme_subjects']) != 4 or "English Language" not in row['utme_subjects']:
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Must have exactly 4 UTME subjects including English Language"})
                elif row['preferred_course'] not in course_names:
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid preferred course"})
                elif row['state_of_origin'] not in NIGERIAN_STATES:
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid state of origin"})
                elif row['gender'] not in ["Male", "Female"]:
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid gender"})
                elif row['learning_style'] not in learning_styles:
                    invalid_rows.append({"row": idx + 2, "student_id": student_id, "error": "Invalid learning style"})

            # Filter out rows with validation errors
            invalid_indices = [r['row'] - 2 for r in invalid_rows]
            valid_df = valid_df[~valid_df.index.isin(invalid_indices)]

            if valid_df.empty:
                st.error("No valid rows to process after validation checks.")
                invalid_csv_buffer = io.StringIO()
                pd.DataFrame(invalid_rows).to_csv(invalid_csv_buffer, index=False)
                st.download_button(
                    label="Download Invalid Rows Report",
                    data=invalid_csv_buffer.getvalue(),
                    file_name="invalid_rows.csv",
                    mime="text/csv",
                    key="invalid_rows_download_validation"
                )
                st.session_state.uploader_key += 1
                st.rerun()

            progress_bar = st.progress(0)
            status_text = st.empty()

            async def update_progress(progress):
                progress_bar.progress(min(1.0, progress))
                status_text.text(f"Processing: {int(progress * 100)}% complete")

            # Process valid CSV applications
            result = await process_with_timeout(
                process_csv_applications(valid_df, course_capacities, jamb_data, neco_data, update_progress),
                timeout=300
            )

            # Ensure result is a tuple
            if not isinstance(result, tuple) or len(result) != 3:
                logger.error("process_csv_applications returned invalid result: %s", result)
                st.error("Processing failed: Invalid return type from process_csv_applications.")
                st.session_state.uploader_key += 1
                st.rerun()

            admission_results, eligible_courses_df, processing_invalid_rows = result
            # Combine invalid rows from preprocessing and processing
            invalid_rows.extend(processing_invalid_rows)

            # Initialize session state
            st.session_state.batch_results = admission_results or []
            st.session_state.eligible_courses_df = eligible_courses_df if not eligible_courses_df.empty else pd.DataFrame()
            st.session_state.invalid_rows = invalid_rows or []

            if admission_results:
                st.success(f"Processed {len(admission_results)} applications successfully!")
                detailed_results, summary_stats, course_breakdown = create_comprehensive_admission_report(
                    admission_results, valid_df, course_capacities
                )

                st.subheader("Summary Statistics")
                for key, value in summary_stats.items():
                    st.write(f"**{key}**: {value:.2f}%")

                st.subheader("Course Breakdown")
                st.dataframe(
                    course_breakdown,
                    use_container_width=True,
                    column_config={
                        'admitted_course': "Course",
                        'Students_Admitted': "Students Admitted",
                        'Avg_Score': st.column_config.NumberColumn("Average Score", format="%.2f"),
                        'Min_Score': st.column_config.NumberColumn("Min Score", format="%.2f"),
                        'Max_Score': st.column_config.NumberColumn("Max Score", format="%.2f"),
                        'Capacity': "Capacity",
                        'Utilization_Rate': st.column_config.NumberColumn("Utilization Rate", format="%.2f%")
                    }
                )

                st.subheader("Detailed Results")
                st.dataframe(
                    detailed_results,
                    use_container_width=True,
                    column_config={
                        'student_id': "Student ID",
                        'name': "Name",
                        'admitted_course': "Admitted Course",
                        'status': "Status",
                        'admission_type': "Admission Type",
                        'score': st.column_config.NumberColumn("Score", format="%.2f"),
                        'rank': "Rank",
                        'reason': "Reason",
                        'original_preference': "Original Preference",
                        'recommendation_reason': "Recommendation Reason",
                        'available_alternatives': "Available Alternatives",
                        'suggested_alternatives': "Suggested Alternatives"
                    }
                )

                # Download button for detailed results
                csv_buffer = io.StringIO()
                detailed_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Detailed Results",
                    data=csv_buffer.getvalue(),
                    file_name="batch_admission_results.csv",
                    mime="text/csv",
                    key="detailed_results_download"
                )

                if not eligible_courses_df.empty:
                    st.subheader("Eligible Courses for All Applicants")
                    st.dataframe(
                        eligible_courses_df,
                        use_container_width=True,
                        column_config={
                            'student_id': "Student ID",
                            'name': "Name",
                            'course': "Course",
                            'score': st.column_config.NumberColumn("Score", format="%.2f"),
                            'interest_score': st.column_config.NumberColumn("Interest Score", format="%.2f"),
                            'learning_style_score': st.column_config.NumberColumn("Learning Style Score", format="%.2f"),
                            'diversity_score': st.column_config.NumberColumn("Diversity Score", format="%.2f"),
                            'olevel_pass_rate': st.column_config.NumberColumn("O'Level Pass Rate", format="%.2f")
                        }
                    )
                    eligible_csv_buffer = io.StringIO()
                    eligible_courses_df.to_csv(eligible_csv_buffer, index=False)
                    st.download_button(
                        label="Download Eligible Courses",
                        data=eligible_csv_buffer.getvalue(),
                        file_name="eligible_courses.csv",
                        mime="text/csv",
                        key="eligible_courses_download"
                    )

                if invalid_rows:
                    st.subheader("Invalid Rows")
                    invalid_df = pd.DataFrame(invalid_rows)
                    st.dataframe(
                        invalid_df,
                        use_container_width=True,
                        column_config={
                            'row': "Row Number",
                            'student_id': "Student ID",
                            'error': "Error Message"
                        }
                    )
                    invalid_csv_buffer = io.StringIO()
                    invalid_df.to_csv(invalid_csv_buffer, index=False)
                    st.download_button(
                        label="Download Invalid Rows Report",
                        data=invalid_csv_buffer.getvalue(),
                        file_name="invalid_rows.csv",
                        mime="text/csv",
                        key="invalid_rows_download"
                    )

            else:
                st.error("No valid applications processed. Check the invalid rows report.")
                if invalid_rows:
                    invalid_df = pd.DataFrame(invalid_rows)
                    st.dataframe(
                        invalid_df,
                        use_container_width=True,
                        column_config={
                            'row': "Row Number",
                            'student_id': "Student ID",
                            'error': "Error Message"
                        }
                    )
                    invalid_csv_buffer = io.StringIO()
                    invalid_df.to_csv(invalid_csv_buffer, index=False)
                    st.download_button(
                        label="Download Invalid Rows Report",
                        data=invalid_csv_buffer.getvalue(),
                        file_name="invalid_rows.csv",
                        mime="text/csv",
                        key="invalid_rows_download_2"
                    )

            st.session_state.uploader_key += 1
            st.rerun()

        except Exception as e:
            logger.error("Error processing CSV: %s", str(e))
            st.error(f"Error processing CSV: {str(e)}")
            st.session_state.uploader_key += 1
            st.rerun()

    with tab3:
        st.header("Analytics & Insights")
        st.markdown("Explore admission trends, course demand, and capacity utilization with interactive visualizations.")

        admission_results = st.session_state.batch_results or []
        demand = analyze_student_demand(pd.DataFrame(), jamb_data, neco_data)
        capacity_optimization = optimize_course_capacities(demand, course_capacities)
        capacity_utilization = calculate_capacity_utilization(admission_results, course_capacities)

        # Custom color scheme for consistency (modern and vibrant)
        color_scheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Custom Plotly template for better aesthetics
        custom_template = {
            "layout": {
                "font": {"family": "Arial", "size": 14, "color": "#333333"},
                "paper_bgcolor": "rgba(255,255,255,0.9)",
                "plot_bgcolor": "rgba(245,245,245,0.9)",
                "margin": {"t": 80, "b": 80, "l": 60, "r": 60},
                "showlegend": True,
                "legend": {"x": 1, "y": 1, "xanchor": "right", "yanchor": "top", "font": {"size": 12}},
                "hoverlabel": {"bgcolor": "white", "font_size": 12, "font_family": "Arial"}
            }
        }

        # Layout: Two columns for charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Admitted Students by Faculty (Bar)")
            if admission_results:
                faculty_counts = pd.DataFrame([
                    {
                        'Faculty': get_course_requirements()[course]['faculty'],
                        'Admitted': sum(1 for r in admission_results if r['admitted_course'] == course)
                    } for course in course_names
                ]).groupby('Faculty').sum().reset_index()
                if not faculty_counts.empty:
                    fig = px.bar(
                        faculty_counts, 
                        x='Faculty', 
                        y='Admitted', 
                        title="Admitted Students by Faculty",
                        color='Faculty',
                        color_discrete_sequence=color_scheme,
                        text='Admitted',
                        template=custom_template
                    )
                    fig.update_layout(
                        xaxis_title="Faculty",
                        yaxis_title="Number of Students Admitted",
                        showlegend=False,
                        title_x=0.5,
                        height=400,
                        hovermode="x unified",
                        xaxis_tickangle=45
                    )
                    fig.update_traces(
                        textposition='outside',
                        texttemplate='%{text}',
                        hovertemplate="%{x}<br>Admitted: %{y}<extra></extra>",
                        marker_line_width=1.5,
                        marker_line_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No faculty data available for visualization.")
            else:
                st.warning("No admission results available. Process batch applications in Tab 2 to generate data.")

        with col2:
            st.subheader("Faculty Admission Distribution (Pie)")
            if admission_results:
                faculty_counts = pd.DataFrame([
                    {
                        'Faculty': get_course_requirements()[course]['faculty'],
                        'Admitted': sum(1 for r in admission_results if r['admitted_course'] == course)
                    } for course in course_names
                ]).groupby('Faculty').sum().reset_index()
                if not faculty_counts.empty:
                    fig = px.pie(
                        faculty_counts, 
                        names='Faculty', 
                        values='Admitted', 
                        title="Distribution of Admitted Students by Faculty",
                        color_discrete_sequence=color_scheme,
                        template=custom_template
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        textposition='inside',
                        hovertemplate="%{label}<br>Admitted: %{value}<br>Percentage: %{percent}<extra></extra>",
                        marker=dict(line=dict(color='white', width=1.5))
                    )
                    fig.update_layout(
                        title_x=0.5,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No faculty data available for visualization.")
            else:
                st.warning("No admission results available for visualization.")

        st.subheader("NECO Pass Rates by State and Gender")
        if not neco_data.empty:
            neco_melted = neco_data.melt(id_vars=['state'], value_vars=['male_pass_rate', 'female_pass_rate'], 
                                        var_name='Gender', value_name='Pass Rate')
            neco_melted['Gender'] = neco_melted['Gender'].replace({'male_pass_rate': 'Male', 'female_pass_rate': 'Female'})
            if not neco_melted.empty:
                fig = px.bar(
                    neco_melted, 
                    x='state', 
                    y='Pass Rate', 
                    color='Gender', 
                    barmode='group', 
                    title="NECO Pass Rates by State and Gender",
                    color_discrete_sequence=color_scheme[:2],
                    template=custom_template
                )
                fig.update_layout(
                    xaxis_title="State",
                    yaxis_title="Pass Rate (%)",
                    xaxis_tickangle=45,
                    title_x=0.5,
                    height=450,
                    legend_title="Gender",
                    hovermode="x unified"
                )
                fig.update_traces(
                    hovertemplate="%{x}<br>%{fullData.name}<br>Pass Rate: %{y:.2f}%<extra></extra>",
                    marker_line_width=1.5,
                    marker_line_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No NECO data available for visualization.")
        else:
            st.warning("No NECO data available for visualization.")

        st.subheader("Course Demand by Faculty (Heatmap)")
        demand_df = pd.DataFrame([
            {
                'Course': course,
                'Faculty': get_course_requirements()[course]['faculty'],
                'Demand Score': data['demand_score'] * 100
            } for course, data in demand.items()
        ])
        if not demand_df.empty:
            demand_pivot = demand_df.pivot_table(
                values='Demand Score', 
                index='Faculty', 
                columns='Course', 
                aggfunc='sum', 
                fill_value=0
            )
            fig = px.imshow(
                demand_pivot,
                title="Course Demand by Faculty",
                color_continuous_scale='Viridis',
                labels=dict(x="Course", y="Faculty", color="Demand Score (%)"),
                template=custom_template
            )
            fig.update_layout(
                xaxis_tickangle=45,
                title_x=0.5,
                height=500,
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12))
            )
            fig.update_traces(
                hovertemplate="Faculty: %{y}<br>Course: %{x}<br>Demand Score: %{z:.2f}%<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No demand data available for visualization.")

        st.subheader("Admission Scores by Course (Line)")
        if admission_results:
            course_scores = pd.DataFrame([
                {
                    'Course': r['admitted_course'],
                    'Average Score': r['score']
                } for r in admission_results if r['status'] == "Admitted"
            ])
            if not course_scores.empty:
                course_scores = course_scores.groupby('Course').mean().reset_index()
                fig = px.line(
                    course_scores, 
                    x='Course', 
                    y='Average Score', 
                    title="Average Admission Scores by Course",
                    markers=True,
                    color_discrete_sequence=color_scheme,
                    template=custom_template
                )
                fig.update_layout(
                    xaxis_title="Course",
                    yaxis_title="Average Prediction Score",
                    xaxis_tickangle=45,
                    title_x=0.5,
                    height=400,
                    hovermode="x unified"
                )
                fig.update_traces(
                    hovertemplate="Course: %{x}<br>Average Score: %{y:.2f}<extra></extra>",
                    line=dict(width=2.5),
                    marker=dict(size=8)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No admission scores available for visualization.")
        else:
            st.warning("No admission results available. Process batch applications in Tab 2 to generate data.")

        st.subheader("Faculty Capacity Utilization")
        if admission_results:
            utilization_df = pd.DataFrame([
                {
                    'Course': course,
                    'Utilization Rate': data['Utilization Rate'],
                    'Status': data['Status']
                } for course, data in capacity_utilization.items()
            ])
            if not utilization_df.empty:
                fig = px.bar(
                    utilization_df, 
                    x='Course', 
                    y='Utilization Rate', 
                    color='Status', 
                    title="Course Capacity Utilization",
                    color_discrete_map={'Optimal': '#00CC96', 'Overcapacity': '#EF553B'},
                    text='Utilization Rate',
                    template=custom_template
                )
                fig.update_layout(
                    xaxis_title="Course",
                    yaxis_title="Utilization Rate (%)",
                    xaxis_tickangle=45,
                    title_x=0.5,
                    height=450,
                    legend_title="Status",
                    hovermode="x unified"
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    hovertemplate="Course: %{x}<br>Utilization: %{y:.1f}%<br>Status: %{fullData.name}<extra></extra>",
                    marker_line_width=1.5,
                    marker_line_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No utilization data available for visualization.")
        else:
            st.warning("No admission results available for visualization.")

        with st.expander("Capacity Optimization Suggestions", expanded=False):
            st.markdown("Recommendations for adjusting course capacities based on demand.")
            optimization_df = pd.DataFrame([
                {
                    'Course': course,
                    'Current Capacity': data['current_capacity'],
                    'Suggested Capacity': data['suggested_capacity'],
                    'Priority': data['priority']
                } for course, data in capacity_optimization.items()
            ])
            if not optimization_df.empty:
                st.dataframe(
                    optimization_df,
                    use_container_width=True,
                    column_config={
                        'Course': "Course",
                        'Current Capacity': "Current Capacity",
                        'Suggested Capacity': "Suggested Capacity",
                        'Priority': "Priority"
                    }
                )
            else:
                st.warning("No optimization data available.")
    with tab4:
            st.header("Help & FAQ")
            st.markdown("""
            ### Frequently Asked Questions
            **Q: What is this system?**
            A: This is an ML-based admission prediction system for FUTA, using JAMB 2017-2018 and NECO 2016 data to predict eligibility and placement for 47 courses.

            **Q: How do I use the Individual Admission tab?**
            A: Enter your details, including UTME score, subjects (must include English Language), O'Level grades (at least 5 subjects), interests, and learning style. Submit to get a prediction and download an admission letter if eligible.

            **Q: What format should the CSV file have for batch processing?**
            A: The CSV must include the following columns:
            - `student_id`: Unique identifier
            - `name`: Full name
            - `utme_score`: UTME score (0-400)
            - `preferred_course`: Chosen course (must match one of the 47 FUTA courses)
            - `utme_subjects`: Comma-separated list of 4 subjects (including English Language)
            - `interests`: Comma-separated list of interests
            - `learning_style`: One of Analytical Thinker, Visual Learner, Practical Learner, Conceptual Learner, Social Learner
            - `state_of_origin`: One of the 37 Nigerian states
            - `gender`: Male or Female
            - Subject grades (e.g., `english_language_grade`, `mathematics_grade`) with values A1, B2, B3, C4, C5, or C6

            **Sample CSV:**
            ```csv
            student_id,name,utme_score,preferred_course,utme_subjects,interests,learning_style,state_of_origin,gender,english_language_grade,mathematics_grade,physics_grade,chemistry_grade,biology_grade
            001,John Doe,250,Biochemistry,English Language,Physics,Chemistry,Biology,Health Sciences,Analytical Thinker,Lagos,Male,A1,B3,C4,C4,C5
            ```

            **Q: How are predictions made?**
            A: The system uses a Random Forest model trained on synthetic data, combined with rule-based eligibility checks, considering UTME scores, O'Level grades, interests, learning styles, and diversity factors.

            **Q: What if I encounter an error?**
            A: Check the error message for details. For batch processing, download the invalid rows report to identify issues. Ensure all required fields are correctly formatted, including English Language in UTME subjects.

            **Q: Can I reset the form?**
            A: Yes, use the 'Reset Form' button in the Individual Admission tab to clear inputs and start over.

            **Q: How are course capacities determined?**
            A: Capacities are dynamically calculated based on JAMB faculty demand, with a minimum of 50 slots per course, adjusted by faculty admission proportions.
            """)

async def main_async():
    await main()

if __name__ == "__main__":
    asyncio.run(main_async())
