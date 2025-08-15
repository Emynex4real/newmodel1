import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import base64
from datetime import datetime
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FUTA Admission Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global constants
COURSE_CAPACITIES = {
    "Computer Science": 200,
    "Civil Engineering": 150,
    "Electrical / Electronics Engineering": 120,
    "Mechanical Engineering": 100,
    "Software Engineering": 80,
    "Cyber Security": 60,
    "Information Systems": 70,
    "Information Technology": 90,
    "Computer Engineering": 80,
    "Industrial & Production Engineering": 60,
    "Architecture": 50,
    "Building": 40,
    "Human Anatomy": 100,
    "Medical Laboratory Science": 80,
    "Physiology": 70,
    "Biochemistry": 60,
    "Mathematics": 80,
    "Physics": 70,
    "Statistics": 50,
    "Industrial Mathematics": 40,
}

DEFAULT_CAPACITY = 50
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
    "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara"
]

@st.cache_data
def load_course_data():
    """Load and return all course-related data"""
    logger.info("Loading course data")
    
    common_subjects = [
        "English Language", "Mathematics", "Physics", "Chemistry", "Biology",
        "Agricultural Science", "Geography", "Economics", "Further Mathematics",
        "Statistics", "Fine Art", "Technical Drawing", "Introduction to Building Construction",
        "Government", "Commerce", "Accounting", "Literature in English", "History",
        "CRK", "IRK", "Social Studies"
    ]

    grade_map = {
        "A1": 1, "B2": 2, "B3": 3, "C4": 4, "C5": 5, "C6": 6, "D7": 7, "E8": 8, "F9": 9
    }

    course_names = [
        "Agric Extension & Communication Technology", "Agricultural Engineering",
        "Agriculture Resource Economics", "Animal Production & Health Services",
        "Applied Geology", "Applied Geophysics", "Architecture", "Biochemistry",
        "Biology", "Biomedical Technology", "Biotechnology", "Building",
        "Civil Engineering", "Computer Engineering", "Computer Science",
        "Crop Soil & Pest Management", "Cyber Security", "Ecotourism & Wildlife Management",
        "Electrical / Electronics Engineering", "Entrepreneurship", "Estate Management",
        "Fisheries & Aquaculture", "Food Science & Technology", "Forestry & Wood Technology",
        "Human Anatomy", "Industrial & Production Engineering", "Industrial Chemistry",
        "Industrial Design", "Industrial Mathematics", "Information & Communication Technology",
        "Information Systems", "Information Technology", "Marine Science & Technology",
        "Mathematics", "Mechanical Engineering", "Medical Laboratory Science",
        "Metallurgical & Materials Engineering", "Meteorology", "Microbiology",
        "Mining Engineering", "Physics", "Physiology", "Quantity Surveying",
        "Remote Sensing & Geoscience Information System", "Software Engineering",
        "Statistics", "Surveying & Geoinformatics", "Textile Design Technology",
        "Urban & Regional Planning"
    ]

    course_groups = {
        "agriculture": [
            "Agric Extension & Communication Technology", "Agricultural Engineering",
            "Agriculture Resource Economics", "Animal Production & Health Services",
            "Crop Soil & Pest Management", "Ecotourism & Wildlife Management",
            "Fisheries & Aquaculture", "Food Science & Technology", "Forestry & Wood Technology"
        ],
        "engineering": [
            "Agricultural Engineering", "Civil Engineering", "Computer Engineering",
            "Electrical / Electronics Engineering", "Industrial & Production Engineering",
            "Mechanical Engineering", "Metallurgical & Materials Engineering", "Mining Engineering"
        ],
        "science": [
            "Applied Geology", "Applied Geophysics", "Biochemistry", "Biology",
            "Biomedical Technology", "Biotechnology", "Industrial Chemistry",
            "Industrial Mathematics", "Marine Science & Technology", "Mathematics",
            "Meteorology", "Microbiology", "Physics", "Statistics"
        ],
        "technology": [
            "Architecture", "Building", "Computer Science", "Cyber Security",
            "Information & Communication Technology", "Information Systems",
            "Information Technology", "Software Engineering"
        ],
        "health": ["Human Anatomy", "Medical Laboratory Science", "Physiology"],
        "management": [
            "Entrepreneurship", "Estate Management", "Quantity Surveying",
            "Surveying & Geoinformatics", "Urban & Regional Planning"
        ],
        "design": ["Industrial Design", "Textile Design Technology"],
        "geoscience": ["Remote Sensing & Geoscience Information System"]
    }

    course_details = {
        "Computer Science": {
            "description": "Study of computational systems, algorithms, and software design",
            "duration": "4 years",
            "career_paths": ["Software Developer", "Data Scientist", "Systems Analyst", "IT Consultant"],
            "average_salary": "₦2,500,000 - ₦8,000,000",
            "job_outlook": "Excellent",
            "skills_developed": ["Programming", "Problem Solving", "Data Analysis", "System Design"]
        },
        "Civil Engineering": {
            "description": "Design and construction of infrastructure projects",
            "duration": "5 years",
            "career_paths": ["Structural Engineer", "Project Manager", "Construction Manager", "Urban Planner"],
            "average_salary": "₦2,000,000 - ₦6,000,000",
            "job_outlook": "Very Good",
            "skills_developed": ["Technical Design", "Project Management", "Problem Solving", "Leadership"]
        },
        "Human Anatomy": {
            "description": "Study of human body structure and function",
            "duration": "6 years",
            "career_paths": ["Medical Doctor", "Anatomist", "Medical Researcher"],
            "average_salary": "₦3,000,000 - ₦15,000,000",
            "job_outlook": "Excellent",
            "skills_developed": ["Critical Thinking", "Scientific Analysis", "Research"]
        }
    }

    interest_categories = {
        "Problem Solving & Logic": ["Computer Science", "Mathematics", "Software Engineering", "Cyber Security"],
        "Building & Construction": ["Civil Engineering", "Architecture", "Building", "Quantity Surveying"],
        "Healthcare & Medicine": ["Human Anatomy", "Medical Laboratory Science", "Physiology", "Biochemistry"],
        "Environment & Nature": ["Agriculture Resource Economics", "Forestry & Wood Technology", "Ecotourism & Wildlife Management"],
        "Technology & Innovation": ["Computer Engineering", "Information Technology", "Electrical / Electronics Engineering"],
        "Business & Management": ["Entrepreneurship", "Estate Management", "Urban & Regional Planning"],
        "Research & Analysis": ["Applied Geology", "Statistics", "Biotechnology", "Microbiology"],
        "Creative & Design": ["Industrial Design", "Textile Design Technology", "Architecture"]
    }

    learning_styles = {
        "Visual Learner": ["Architecture", "Industrial Design", "Computer Science", "Mathematics"],
        "Hands-on Learner": ["Civil Engineering", "Mechanical Engineering", "Building", "Agricultural Engineering"],
        "Analytical Thinker": ["Mathematics", "Statistics", "Computer Science", "Physics"],
        "People-oriented": ["Human Anatomy", "Entrepreneurship", "Agric Extension & Communication Technology", "Estate Management"]
    }

    cutoff_marks = {
        "Human Anatomy": 250,
        "Medical Laboratory Science": 250,
        "Physiology": 250,
        "Civil Engineering": 220,
        "Computer Engineering": 220,
        "Electrical / Electronics Engineering": 220,
        "Mechanical Engineering": 220,
        "Metallurgical & Materials Engineering": 220,
        "Mining Engineering": 220,
        "Computer Science": 230,
        "Cyber Security": 230,
        "Software Engineering": 230,
    }
    for course in course_names:
        if course not in cutoff_marks:
            cutoff_marks[course] = 180

    return common_subjects, grade_map, course_names, course_groups, cutoff_marks, course_details, interest_categories, learning_styles

common_subjects, grade_map, course_names, course_groups, cutoff_marks, course_details, interest_categories, learning_styles = load_course_data()

# Store feature names globally for consistency
FEATURE_NAMES = None

def get_course_requirements():
    """Return detailed course requirements for all courses"""
    requirements = {
        "Computer Science": {
            "mandatory": ["English Language", "Mathematics", "Physics"],
            "or_groups": [],
            "optional": {2: ["Biology", "Chemistry", "Economics", "Geography"]},
            "thresholds": {},
            "required_credit_count": 5,
        },
        "Civil Engineering": {
            "mandatory": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "or_groups": [],
            "optional": {1: ["Biology", "Further Mathematics", "Technical Drawing"]},
            "thresholds": {},
            "required_credit_count": 5,
        },
        "Human Anatomy": {
            "mandatory": ["English Language", "Mathematics", "Biology", "Chemistry", "Physics"],
            "or_groups": [],
            "optional": {},
            "thresholds": {},
            "required_credit_count": 5,
        },
        "Software Engineering": {
            "mandatory": ["English Language", "Mathematics", "Physics"],
            "or_groups": [],
            "optional": {2: ["Chemistry", "Further Mathematics", "Economics"]},
            "thresholds": {},
            "required_credit_count": 5,
        },
        "Cyber Security": {
            "mandatory": ["English Language", "Mathematics", "Physics"],
            "or_groups": [],
            "optional": {2: ["Chemistry", "Economics", "Further Mathematics"]},
            "thresholds": {},
            "required_credit_count": 5,
        },
        "Information Systems": {
            "mandatory": ["English Language", "Mathematics"],
            "or_groups": [],
            "optional": {3: ["Physics", "Chemistry", "Economics", "Geography"]},
            "thresholds": {},
            "required_credit_count": 5,
        },
    }
    default_req = {
        "mandatory": ["English Language", "Mathematics"],
        "or_groups": [],
        "optional": {3: common_subjects[2:]},
        "thresholds": {},
        "required_credit_count": 5,
    }
    for course in course_names:
        if course not in requirements:
            requirements[course] = default_req
    return requirements

@st.cache_resource
def train_placement_model():
    """Train a lightweight ML model for placement prediction"""
    global FEATURE_NAMES
    logger.info("Starting placement model training")
    try:
        parsed_req = get_course_requirements()
        data = []
        
        # Minimal dataset size for Streamlit Cloud
        for _ in range(500):  # Reduced to 500 for faster training
            utme = np.random.randint(100, 400)
            num_subjects = np.random.randint(5, 10)
            selected_subs = np.random.choice(common_subjects, num_subjects, replace=False)
            olevels = {sub: np.random.choice(list(grade_map.values())) for sub in selected_subs}
            interests = np.random.choice(list(interest_categories.keys()), np.random.randint(1, 5), replace=False).tolist()
            learning = np.random.choice(list(learning_styles.keys()))
            state = np.random.choice(NIGERIAN_STATES)
            gender = np.random.choice(["Male", "Female", "Other"])
            
            for course in course_names:
                eligible = is_eligible(olevels, course, parsed_req) and utme >= cutoff_marks[course]
                score = 0
                interest_weight = 0
                diversity_score = 0.5 if state in ["Yobe", "Zamfara", "Borno"] else 0.3 if gender == "Female" else 0
                if eligible:
                    grade_sum, count = compute_grade_sum(olevels, course, parsed_req)
                    interest_weight = sum(1 for int_ in interests if course in interest_categories.get(int_, [])) * 0.3
                    if learning in learning_styles and course in learning_styles[learning]:
                        interest_weight += 0.2
                    score = compute_enhanced_score(utme, grade_sum, count, interest_weight, diversity_score)
                
                features = {
                    'utme': utme,
                    'score': score  # Explicitly add score to features
                }
                for sub in common_subjects:
                    features[sub] = olevels.get(sub, 9)
                for int_ in interest_categories.keys():
                    features[int_] = 1 if int_ in interests else 0
                for ls in learning_styles.keys():
                    features[f'ls_{ls}'] = 1 if learning == ls else 0
                features['diversity_score'] = diversity_score
                features['course'] = course
                data.append(features)
        
        logger.info("Synthetic data generation complete")
        df = pd.DataFrame(data)
        df = pd.get_dummies(df, columns=['course'])
        X = df.drop('score', axis=1)
        y = df['score']
        
        # Store feature names for prediction
        FEATURE_NAMES = X.columns.tolist()
        
        logger.info("Splitting data for training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Simple model without GridSearchCV for speed
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model trained successfully with MSE: {mse:.4f}")
        
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        st.error(f"Failed to train model: {str(e)}")
        raise

def get_dynamic_course_capacities(df):
    """Dynamically adjust course capacities based on demand"""
    demand_analysis = analyze_student_demand(df)
    dynamic_capacities = COURSE_CAPACITIES.copy()
    total_capacity = sum(dynamic_capacities.values())
    total_demand = sum(v['total_estimated_demand'] for v in demand_analysis.values())
    
    for course in course_names:
        if course not in dynamic_capacities:
            dynamic_capacities[course] = DEFAULT_CAPACITY
        demand = demand_analysis.get(course, {}).get('total_estimated_demand', 0)
        if demand > dynamic_capacities[course] * 1.5 and total_capacity > 0:
            increase = min(int(demand * 0.2), total_capacity // len(course_names))
            dynamic_capacities[course] += increase
    return dynamic_capacities

def calculate_capacity_utilization(admission_results, course_capacities):
    """Calculate capacity utilization statistics"""
    utilization_stats = {}
    course_admissions = defaultdict(int)
    for result in admission_results:
        if result['status'] in ['ADMITTED', 'ALTERNATIVE_ADMISSION']:
            course_admissions[result['admitted_course']] += 1
    for course, capacity in course_capacities.items():
        admitted = course_admissions[course]
        utilization_rate = (admitted / capacity) * 100 if capacity > 0 else 0
        utilization_stats[course] = {
            'capacity': capacity,
            'admitted': admitted,
            'available': capacity - admitted,
            'utilization_rate': utilization_rate,
            'status': 'Full' if admitted >= capacity else 'Available'
        }
    return utilization_stats

def optimize_course_capacities(student_demand, current_capacities):
    """Suggest optimal course capacity allocation"""
    suggestions = {}
    total_capacity = sum(current_capacities.values())
    for course, demand in student_demand.items():
        current = current_capacities.get(course, DEFAULT_CAPACITY)
        ratio = demand / current if current > 0 else float('inf')
        if ratio > 1.5:
            suggested = min(int(demand * 1.2), total_capacity // len(course_names))
            suggestions[course] = {
                'current': current,
                'suggested': suggested,
                'reason': f"High demand ({demand} vs {current})",
                'priority': 'High'
            }
        elif ratio < 0.5:
            suggested = max(int(demand * 1.5), 20)
            suggestions[course] = {
                'current': current,
                'suggested': suggested,
                'reason': f"Low demand ({demand} vs {current})",
                'priority': 'Medium'
            }
    return suggestions

def analyze_student_demand(df):
    """Analyze student demand patterns"""
    demand_analysis = {}
    if 'preferred_course' in df.columns:
        preferred_counts = df['preferred_course'].value_counts().to_dict()
        interest_course_mapping = defaultdict(int)
        if 'interests' in df.columns:
            for _, row in df.iterrows():
                interests = str(row.get('interests', '')).split(',')
                for interest in interests:
                    interest = interest.strip()
                    if interest in interest_categories:
                        for course in interest_categories[interest]:
                            interest_course_mapping[course] += 1
        for course in course_names:
            primary = preferred_counts.get(course, 0)
            secondary = interest_course_mapping.get(course, 0) * 0.3
            total = int(primary + secondary)
            demand_analysis[course] = {
                'primary_demand': primary,
                'secondary_demand': int(secondary),
                'total_estimated_demand': total,
                'demand_category': 'High' if total > 100 else 'Medium' if total > 50 else 'Low'
            }
    return demand_analysis

def is_eligible(applicant_subjects, course, parsed_req):
    """Check if applicant meets O'Level requirements"""
    if course not in parsed_req:
        required_subjects = ["English Language", "Mathematics"]
        return all(sub in applicant_subjects and applicant_subjects[sub] <= 6 for sub in required_subjects)
    p = parsed_req[course]
    thresh = p["thresholds"]
    good_credit = {s for s in applicant_subjects if applicant_subjects[s] <= 6}
    if len(good_credit) < p["required_credit_count"]:
        return False
    for s in p["mandatory"]:
        if s not in applicant_subjects or applicant_subjects[s] > thresh.get(s, 6):
            return False
    for group in p["or_groups"]:
        if not any(sub in applicant_subjects and applicant_subjects[sub] <= thresh.get(sub, 6) for sub in group):
            return False
    for num, group in p["optional"].items():
        count = sum(1 for sub in group if sub in applicant_subjects and applicant_subjects[sub] <= thresh.get(sub, 6))
        if count < num:
            return False
    return True

def compute_grade_sum(applicant_subjects, course, parsed_req):
    """Compute sum of grades and count of subjects"""
    if course not in parsed_req:
        relevant_subjects = ["English Language", "Mathematics", "Physics", "Chemistry", "Biology"]
        grade_sum = sum(applicant_subjects.get(sub, 9) for sub in relevant_subjects)
        count = sum(1 for sub in relevant_subjects if sub in applicant_subjects)
        return grade_sum, max(count, 1)
    p = parsed_req[course]
    thresh = p["thresholds"]
    grade_sum = 0
    count = 0
    for s in p["mandatory"]:
        if s in applicant_subjects and applicant_subjects[s] <= thresh.get(s, 6):
            grade_sum += applicant_subjects[s]
            count += 1
    for group in p["or_groups"]:
        candidates = [applicant_subjects[sub] for sub in group if sub in applicant_subjects and applicant_subjects[sub] <= thresh.get(sub, 6)]
        if candidates:
            grade_sum += min(candidates)
            count += 1
    for num, group in p["optional"].items():
        candidates = [applicant_subjects[sub] for sub in group if sub in applicant_subjects and applicant_subjects[sub] <= thresh.get(sub, 6)]
        candidates.sort()
        if len(candidates) >= num:
            grade_sum += sum(candidates[:num])
            count += num
    return grade_sum, max(count, 1)

def compute_enhanced_score(utme_score, grade_sum, count, course_weight, diversity_score, base_interest_bonus=0.15):
    """Compute enhanced score with diversity factor"""
    if count == 0:
        return 0
    normalized_utme = utme_score / 400
    average_grade = grade_sum / count
    normalized_grade = (9 - average_grade) / 8
    base_score = 0.4 * normalized_utme + 0.4 * normalized_grade
    interest_bonus = course_weight * base_interest_bonus
    diversity_bonus = diversity_score * 0.1
    return min(base_score + interest_bonus + diversity_bonus, 1 + base_interest_bonus + 0.1)

def predict_placement_enhanced(utme_score, olevel_subjects, selected_interests, learning_style, state, gender):
    """Predict placement using ML model"""
    global FEATURE_NAMES
    try:
        model, scaler = train_placement_model()
        if FEATURE_NAMES is None:
            raise ValueError("Feature names not initialized. Please ensure model training is complete.")
        
        parsed_req = get_course_requirements()
        results = []
        for course in course_names:
            eligible = is_eligible(olevel_subjects, course, parsed_req) and utme_score >= cutoff_marks[course]
            interest_weight = sum(1 for int_ in selected_interests if course in interest_categories.get(int_, [])) * 0.3
            if learning_style in learning_styles and course in learning_styles[learning_style]:
                interest_weight += 0.2
            diversity_score = 0.5 if state in ["Yobe", "Zamfara", "Borno"] else 0.3 if gender == "Female" else 0
            
            # Create feature dictionary
            features = {'utme': utme_score}
            for sub in common_subjects:
                features[sub] = olevel_subjects.get(sub, 9)
            for int_ in interest_categories.keys():
                features[int_] = 1 if int_ in selected_interests else 0
            for ls in learning_styles.keys():
                features[f'ls_{ls}'] = 1 if learning_style == ls else 0
            features['diversity_score'] = diversity_score
            for c in course_names:
                features[f'course_{c}'] = 1 if c == course else 0
            
            # Ensure feature alignment
            features_df = pd.DataFrame([features])
            # Add missing features with default value 0
            for feature in FEATURE_NAMES:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            # Drop extra features not in training
            features_df = features_df[FEATURE_NAMES]
            
            X_scaled = scaler.transform(features_df)
            score = model.predict(X_scaled)[0]
            if eligible:
                grade_sum, count = compute_grade_sum(olevel_subjects, course, parsed_req)
                score = compute_enhanced_score(utme_score, grade_sum, count, interest_weight, diversity_score)
            
            results.append({
                "course": course,
                "eligible": eligible,
                "score": score,
                "interest_weight": interest_weight,
                "diversity_score": diversity_score
            })
        
        results_df = pd.DataFrame(results)
        eligible_courses = results_df[results_df["eligible"] & (results_df["score"] > 0)]
        if eligible_courses.empty:
            # Improved error message with specific reasons
            reasons = []
            if utme_score < 160:
                reasons.append(f"Your UTME score ({utme_score}) is below the minimum university requirement of 160.")
            if len([s for s, g in olevel_subjects.items() if g <= 6]) < 5:
                reasons.append("You need at least 5 O'Level credits (C6 or better).")
            if "English Language" not in olevel_subjects or olevel_subjects.get("English Language", 9) > 6:
                reasons.append("A credit in English Language (C6 or better) is required for all courses.")
            if "Mathematics" not in olevel_subjects or olevel_subjects.get("Mathematics", 9) > 6:
                reasons.append("A credit in Mathematics (C6 or better) is required for all courses.")
            
            reason_text = "You are not eligible for any course due to the following reasons:\n" + "\n".join(f"- {r}" for r in reasons)
            if not reasons:
                reason_text = "You are not eligible for any course. Please check your UTME score and O'Level grades against course requirements."
            
            suggestions = [
                "📈 Retake UTME to improve your score if it is below the required cutoff.",
                "📚 Ensure at least 5 O'Level credits, including English Language and Mathematics.",
                "🔍 Review specific course requirements in the university guidelines.",
                "🎯 Consider courses with lower UTME cutoffs (e.g., 180).",
                "📞 Contact the admissions office for guidance."
            ]
            
            return {
                "predicted_program": "UNASSIGNED",
                "score": 0,
                "reason": reason_text,
                "suggestions": suggestions,
                "all_eligible": pd.DataFrame()
            }
        best_course = eligible_courses.loc[eligible_courses["score"].idxmax()]
        return {
            "predicted_program": best_course["course"],
            "score": best_course["score"],
            "reason": "Best match based on ML prediction",
            "all_eligible": eligible_courses.sort_values("score", ascending=False),
            "interest_alignment": best_course["interest_weight"] > 0
        }
    except Exception as e:
        logger.error(f"Error in predict_placement_enhanced: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
        return {
            "predicted_program": "UNASSIGNED",
            "score": 0,
            "reason": f"Prediction failed due to an unexpected error: {str(e)}",
            "all_eligible": pd.DataFrame()
        }

def validate_csv(df):
    """Validate CSV input data"""
    required_columns = ['student_id', 'name', 'utme_score', 'preferred_course']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            st.error(f"CSV file must contain {col} column")
            return False
    df['utme_score'] = pd.to_numeric(df['utme_score'], errors='coerce')
    if df['utme_score'].isna().any():
        logger.error("Invalid UTME scores detected")
        st.error("UTME scores must be numeric")
        return False
    for subject in common_subjects:
        grade_col = f"{subject.lower().replace(' ', '_').replace('/', '_')}_grade"
        if grade_col in df.columns:
            invalid_grades = df[grade_col].notna() & ~df[grade_col].isin(grade_map.keys())
            if invalid_grades.any():
                logger.error(f"Invalid grades in {grade_col}")
                st.error(f"Invalid grades found in {grade_col}. Must be one of {list(grade_map.keys())}")
                return False
    return True

def process_csv_applications(df, course_capacities):
    """Process batch applications with validation"""
    logger.info("Processing CSV applications")
    try:
        if not validate_csv(df):
            return []
        processed_students = []
        for idx, row in df.iterrows():
            student_data = {
                'student_id': str(row.get('student_id', f'STU_{idx:04d}')),
                'name': str(row.get('name', f'Student_{idx}')),
                'utme_score': float(row.get('utme_score', 0)),
                'preferred_course': str(row.get('preferred_course', '')),
                'interests': str(row.get('interests', '')).split(',') if pd.notna(row.get('interests')) else [],
                'learning_style': str(row.get('learning_style', 'Analytical Thinker')),
                'olevel_subjects': {},
                'state_of_origin': str(row.get('state_of_origin', '')),
                'gender': str(row.get('gender', '')),
                'age': int(row.get('age', 18))
            }
            for subject in common_subjects:
                grade_col = f"{subject.lower().replace(' ', '_').replace('/', '_')}_grade"
                if grade_col in row and pd.notna(row[grade_col]):
                    student_data['olevel_subjects'][subject] = grade_map.get(str(row[grade_col]), 9)
            processed_students.append(student_data)
        admission_results = run_intelligent_admission_algorithm_v2(processed_students, course_capacities)
        return admission_results
    except Exception as e:
        logger.error(f"Error in process_csv_applications: {str(e)}")
        st.error(f"Failed to process CSV: {str(e)}")
        return []

def calculate_comprehensive_score(student, course, base_score):
    """Calculate comprehensive score with diversity"""
    final_score = base_score * 0.6
    interest_bonus = sum(0.05 for interest in student['interests'] if interest.strip() in interest_categories and course in interest_categories[interest.strip()])
    final_score += min(interest_bonus, 0.2)
    learning_bonus = 0.1 if student['learning_style'] in learning_styles and any(course_type in course for course_type in ['Engineering', 'Science', 'Technology']) and student['learning_style'] in ['Analytical Thinker', 'Visual Learner'] else 0
    final_score += learning_bonus
    olevel_performance = calculate_olevel_performance_indicator(student['olevel_subjects'])
    final_score += olevel_performance * 0.1
    diversity_score = 0.05 if student['state_of_origin'] in ["Yobe", "Zamfara", "Borno"] else 0.03 if student['gender'] == "Female" else 0
    final_score += diversity_score
    return min(final_score, 1.0)

def calculate_olevel_performance_indicator(olevel_subjects):
    """Calculate O'Level performance indicator"""
    if not olevel_subjects:
        return 0
    grades = list(olevel_subjects.values())
    excellent_count = sum(1 for grade in grades if grade <= 2)
    good_count = sum(1 for grade in grades if grade <= 4)
    total = len(grades)
    return min((excellent_count * 1.0 + (good_count - excellent_count) * 0.7) / total, 1.0)

def calculate_course_similarity(course1, course2, course_groups):
    """Calculate course similarity"""
    similarity = 0.0
    for group, courses in course_groups.items():
        if course1 in courses and course2 in courses:
            similarity += 0.4
            break
    course1_groups = [g for g, cs in course_groups.items() if course1 in cs]
    course2_groups = [g for g, cs in course_groups.items() if course2 in cs]
    related_groups = {
        'engineering': ['technology', 'science'],
        'technology': ['engineering', 'science'],
        'science': ['engineering', 'technology', 'health'],
        'health': ['science'],
        'management': ['design'],
        'design': ['management']
    }
    for g1 in course1_groups:
        for g2 in course2_groups:
            if g1 != g2 and g2 in related_groups.get(g1, []):
                similarity += 0.2
    course1_words = set(course1.lower().split())
    course2_words = set(course2.lower().split())
    common = course1_words.intersection(course2_words)
    if common:
        similarity += len(common) * 0.1
    return min(similarity, 1.0)

def find_smart_alternatives(student_data, student_course_matrix, course_capacities, course_admission_counts, max_alternatives=5):
    """Find intelligent alternative courses"""
    student_id = student_data['student_id']
    preferred_course = student_data['preferred_course']
    interests = student_data.get('interests', [])
    student_row = next((row for row in student_course_matrix if row['student_id'] == student_id), None)
    if not student_row:
        return []
    alternatives = []
    for course in course_names:
        if course != preferred_course and course in course_capacities and course_admission_counts.get(course, 0) < course_capacities[course] and student_row[course]['eligible']:
            base_score = student_row[course]['score']
            similarity_score = calculate_course_similarity(preferred_course, course, course_groups)
            interest_alignment = sum(0.2 for interest in interests if interest.strip() in interest_categories and course in interest_categories[interest.strip()])
            interest_alignment = min(interest_alignment, 0.6)
            available_slots = course_capacities[course] - course_admission_counts.get(course, 0)
            capacity_factor = min(available_slots / course_capacities[course], 1.0) * 0.1
            demand = course_admission_counts.get(course, 0)
            demand_factor = (1 - (demand / course_capacities[course])) * 0.1 if course_capacities[course] > 0 else 0
            alternative_score = base_score * 0.4 + similarity_score * 0.3 + interest_alignment * 0.2 + capacity_factor * 0.05 + demand_factor * 0.05
            alternatives.append({
                'course': course,
                'score': alternative_score,
                'base_score': base_score,
                'similarity_score': similarity_score,
                'interest_alignment': interest_alignment,
                'available_slots': available_slots,
                'recommendation_reason': generate_alternative_reason(preferred_course, course, similarity_score, interest_alignment)
            })
    alternatives.sort(key=lambda x: x['score'], reverse=True)
    return alternatives[:max_alternatives]

def generate_alternative_reason(preferred_course, alternative_course, similarity_score, interest_alignment):
    """Generate reason for alternative recommendation"""
    reasons = []
    if similarity_score > 0.3:
        reasons.append(f"closely related to {preferred_course}")
    if interest_alignment > 0.3:
        reasons.append("matches your interests")
    if any(keyword in alternative_course.lower() for keyword in ['computer', 'software', 'cyber', 'information']):
        if any(keyword in preferred_course.lower() for keyword in ['computer', 'software', 'cyber', 'information']):
            reasons.append("similar technology focus")
    if any(keyword in alternative_course.lower() for keyword in ['engineering']):
        if any(keyword in preferred_course.lower() for keyword in ['engineering']):
            reasons.append("same engineering discipline")
    if not reasons:
        reasons.append("good academic fit based on your qualifications")
    return ", ".join(reasons)

def run_intelligent_admission_algorithm_v2(students, course_capacities):
    """Advanced admission algorithm with diversity and tie-breaking"""
    logger.info("Running intelligent admission algorithm")
    try:
        results = []
        course_admission_counts = {course: 0 for course in course_capacities}
        student_course_matrix = []
        
        for student in students:
            student_row = {'student_id': student['student_id'], 'student_data': student}
            prediction = predict_placement_enhanced(
                student['utme_score'],
                student['olevel_subjects'],
                student['interests'],
                student['learning_style'],
                student['state_of_origin'],
                student['gender']
            )
            for course in course_names:
                base_score = 0
                eligible = False
                if 'all_eligible' in prediction and not prediction['all_eligible'].empty:
                    course_match = prediction['all_eligible'][prediction['all_eligible']['course'] == course]
                    if not course_match.empty:
                        base_score = course_match.iloc[0]['score']
                        eligible = True
                if eligible:
                    comprehensive_score = calculate_comprehensive_score(student, course, base_score)
                    if course == student['preferred_course']:
                        comprehensive_score *= 1.15
                    student_row[course] = {
                        'score': comprehensive_score,
                        'eligible': True,
                        'base_score': base_score
                    }
                else:
                    student_row[course] = {
                        'score': 0,
                        'eligible': False,
                        'base_score': 0
                    }
            student_course_matrix.append(student_row)
        
        admitted_students = set()
        
        # Phase 1: Preferred course admission with tie-breaking
        for course in course_names:
            if course not in course_capacities:
                continue
            capacity = course_capacities[course]
            candidates = []
            for student_row in student_course_matrix:
                student_id = student_row['student_id']
                student_data = student_row['student_data']
                if student_id not in admitted_students and student_row[course]['eligible'] and student_data['preferred_course'] == course:
                    olevel_avg = sum(student_data['olevel_subjects'].values()) / len(student_data['olevel_subjects']) if student_data['olevel_subjects'] else 9
                    candidates.append({
                        'student_id': student_id,
                        'student_data': student_data,
                        'score': student_row[course]['score'],
                        'olevel_avg': olevel_avg,
                        'course': course
                    })
            candidates.sort(key=lambda x: (x['score'], -x['olevel_avg']), reverse=True)
            for i, candidate in enumerate(candidates[:capacity - course_admission_counts[course]]):
                results.append({
                    'student_id': candidate['student_id'],
                    'admitted_course': course,
                    'status': 'ADMITTED',
                    'score': candidate['score'],
                    'rank': course_admission_counts[course] + i + 1,
                    'admission_type': 'PREFERRED'
                })
                admitted_students.add(candidate['student_id'])
                course_admission_counts[course] += 1
        
        # Phase 2: Merit-based admission
        for course in course_names:
            if course not in course_capacities:
                continue
            remaining = course_capacities[course] - course_admission_counts[course]
            if remaining > 0:
                candidates = []
                for student_row in student_course_matrix:
                    student_id = student_row['student_id']
                    student_data = student_row['student_data']
                    if student_id not in admitted_students and student_row[course]['eligible']:
                        olevel_avg = sum(student_data['olevel_subjects'].values()) / len(student_data['olevel_subjects']) if student_data['olevel_subjects'] else 9
                        candidates.append({
                            'student_id': student_id,
                            'student_data': student_data,
                            'score': student_row[course]['score'],
                            'olevel_avg': olevel_avg,
                            'course': course
                        })
                candidates.sort(key=lambda x: (x['score'], -x['olevel_avg']), reverse=True)
                for i, candidate in enumerate(candidates[:remaining]):
                    results.append({
                        'student_id': candidate['student_id'],
                        'admitted_course': course,
                        'status': 'ADMITTED',
                        'score': candidate['score'],
                        'rank': course_admission_counts[course] + i + 1,
                        'admission_type': 'MERIT'
                    })
                    admitted_students.add(candidate['student_id'])
                    course_admission_counts[course] += 1
        
        # Phase 3: Alternative admissions and waitlisting
        remaining_students = []
        for student_row in student_course_matrix:
            student_id = student_row['student_id']
            student_data = student_row['student_data']
            if student_id not in admitted_students:
                max_alt_score = max([student_row[course]['score'] for course in course_names if student_row[course]['eligible'] and course != student_data['preferred_course']] or [0])
                remaining_students.append((student_row, student_data, max_alt_score))
        remaining_students.sort(key=lambda x: x[2], reverse=True)
        
        for student_row, student_data, _ in remaining_students:
            student_id = student_data['student_id']
            if student_id not in admitted_students:
                smart_alternatives = find_smart_alternatives(
                    student_data, student_course_matrix, course_capacities, course_admission_counts
                )
                admitted = False
                for alternative in smart_alternatives:
                    course = alternative['course']
                    if course_admission_counts.get(course, 0) < course_capacities.get(course, 0):
                        results.append({
                            'student_id': student_id,
                            'admitted_course': course,
                            'status': 'ALTERNATIVE_ADMISSION',
                            'score': alternative['score'],
                            'base_score': alternative['base_score'],
                            'similarity_score': alternative['similarity_score'],
                            'original_preference': student_data['preferred_course'],
                            'admission_type': 'SMART_ALTERNATIVE',
                            'recommendation_reason': alternative['recommendation_reason'],
                            'available_alternatives': len(smart_alternatives)
                        })
                        admitted_students.add(student_id)
                        course_admission_counts[course] += 1
                        admitted = True
                        break
                if not admitted:
                    min_req_met = check_minimum_university_requirements(student_data)
                    status = 'WAITLISTED' if min_req_met and smart_alternatives else 'NOT_QUALIFIED'
                    reason = 'Qualified but no capacity in suitable alternatives' if min_req_met else 'Does not meet minimum requirements'
                    results.append({
                        'student_id': student_id,
                        'admitted_course': 'NONE',
                        'status': status,
                        'score': 0,
                        'reason': reason,
                        'admission_type': status.upper(),
                        'suggested_alternatives': [alt['course'] for alt in smart_alternatives[:3]] if smart_alternatives else []
                    })
        logger.info(f"Admission processing complete: {len(admitted_students)} admitted, {len(students) - len(admitted_students)} remaining")
        return results
    except Exception as e:
        logger.error(f"Error in run_intelligent_admission_algorithm_v2: {str(e)}")
        st.error(f"Admission algorithm failed: {str(e)}")
        return []

def check_minimum_university_requirements(student_data):
    """Check minimum university requirements"""
    utme = student_data.get('utme_score', 0)
    olevels = student_data.get('olevel_subjects', {})
    if utme < 160:
        return False
    required = ['English Language', 'Mathematics']
    credits_count = sum(1 for _, grade in olevels.items() if grade <= 6)
    if credits_count < 5:
        return False
    for req in required:
        if req not in olevels or olevels[req] > 6:
            return False
    return True

def create_comprehensive_admission_report(admission_results, original_df, course_capacities):
    """Create comprehensive admission report"""
    results_df = pd.DataFrame(admission_results)
    detailed_results = results_df.merge(
        original_df[['student_id', 'name', 'utme_score', 'preferred_course']], 
        on='student_id', 
        how='left'
    )
    summary_stats = {
        'Total Applications': len(original_df),
        'Direct Admissions': len(results_df[results_df['status'] == 'ADMITTED']),
        'Alternative Admissions': len(results_df[results_df['status'] == 'ALTERNATIVE_ADMISSION']),
        'Not Qualified': len(results_df[results_df['status'] == 'NOT_QUALIFIED']),
        'Waitlisted': len(results_df[results_df['status'] == 'WAITLISTED']),
        'Overall Admission Rate': f"{((len(results_df[results_df['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]) / len(original_df)) * 100):.1f}%",
        'Processing Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    course_breakdown = results_df[results_df['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])].groupby('admitted_course').agg({
        'student_id': 'count',
        'score': ['mean', 'min', 'max']
    }).round(3)
    course_breakdown.columns = ['Students_Admitted', 'Avg_Score', 'Min_Score', 'Max_Score']
    course_breakdown = course_breakdown.reset_index()
    course_breakdown['Capacity'] = course_breakdown['admitted_course'].map(course_capacities)
    course_breakdown['Utilization_Rate'] = ((course_breakdown['Students_Admitted'] / course_breakdown['Capacity']) * 100).round(1)
    return detailed_results, summary_stats, course_breakdown

def generate_admission_letter(student_data):
    """Generate PDF admission letter"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Federal University of Technology, Akure")
    c.drawString(100, 730, "Admission Office")
    c.drawString(100, 710, datetime.now().strftime('%B %d, %Y'))
    c.drawString(100, 670, f"Dear {student_data['name']},")
    c.drawString(100, 650, "Congratulations on your admission!")
    c.drawString(100, 630, f"You have been offered admission into {student_data['admitted_course']} for the {datetime.now().year}/{datetime.now().year + 1} academic session.")
    c.drawString(100, 610, f"Admission Type: {student_data['admission_type']}")
    if student_data['status'] == 'ALTERNATIVE_ADMISSION':
        c.drawString(100, 590, f"Original Preference: {student_data['original_preference']}")
        c.drawString(100, 570, f"Reason for Alternative: {student_data['recommendation_reason']}")
    c.drawString(100, 550, f"Student ID: {student_data['student_id']}")
    c.drawString(100, 530, "Please proceed to the university portal to accept your offer and complete registration.")
    c.drawString(100, 510, "Sincerely,")
    c.drawString(100, 490, "Admissions Office")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def create_download_button(data, filename, label, file_type="csv"):
    """Create download button"""
    if file_type == "csv":
        if isinstance(data, pd.DataFrame):
            csv_data = data.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">{label}</button></a>'
        else:
            return None
    elif file_type == "excel":
        b64 = base64.b64encode(data.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">{label}</button></a>'
    elif file_type == "json":
        json_data = json.dumps(data, indent=2)
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: #FF9800; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">{label}</button></a>'
    elif file_type == "pdf":
        b64 = base64.b64encode(data.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: #F44336; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">{label}</button></a>'
    else:
        return None
    return href

def display_comprehensive_dashboard(admission_results, original_df, course_capacities):
    """Display comprehensive dashboard"""
    if not admission_results:
        st.warning("No admission results to display. Please check your input data.")
        return
    st.header("📊 Advanced Analytics & Export Center")
    detailed_results, summary_stats, course_breakdown = create_comprehensive_admission_report(admission_results, original_df, course_capacities)
    demand_analysis = analyze_student_demand(original_df)
    capacity_suggestions = optimize_course_capacities({k: v['total_estimated_demand'] for k, v in demand_analysis.items()}, course_capacities)
    
    export_tab1, export_tab2, export_tab3, export_tab4, demand_tab, capacity_tab = st.tabs([
        "📄 Standard Reports", 
        "📊 Excel Analytics", 
        "📋 Admission Letters", 
        "📈 Statistical Analysis",
        "📊 Demand Analysis",
        "🔄 Capacity Suggestions"
    ])
    
    with export_tab1:
        st.subheader("Standard CSV Reports")
        col1, col2, col3 = st.columns(3)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with col1:
            st.markdown("**All Results**")
            all_results_btn = create_download_button(
                detailed_results, 
                f"complete_admission_results_{timestamp}.csv",
                "📥 Download All Results"
            )
            if all_results_btn:
                st.markdown(all_results_btn, unsafe_allow_html=True)
        with col2:
            st.markdown("**Admitted Students**")
            admitted_students = detailed_results[detailed_results['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]
            admitted_btn = create_download_button(
                admitted_students,
                f"admitted_students_{timestamp}.csv",
                "📥 Download Admitted"
            )
            if admitted_btn:
                st.markdown(admitted_btn, unsafe_allow_html=True)
        with col3:
            st.markdown("**Course Analysis**")
            course_btn = create_download_button(
                course_breakdown,
                f"course_analysis_{timestamp}.csv",
                "📥 Download Analysis"
            )
            if course_btn:
                st.markdown(course_btn, unsafe_allow_html=True)
    
    with export_tab2:
        st.subheader("Comprehensive Excel Analytics")
        if st.button("🔄 Generate Excel Report", type="primary"):
            with st.spinner("Generating Excel report..."):
                excel_data = export_to_excel(admission_results, original_df, course_capacities, f"admission_report_{timestamp}.xlsx")
                excel_btn = create_download_button(
                    excel_data,
                    f"comprehensive_admission_report_{timestamp}.xlsx",
                    "📥 Download Excel Report",
                    "excel"
                )
                if excel_btn:
                    st.markdown(excel_btn, unsafe_allow_html=True)
                    st.success("✅ Excel report generated!")
    
    with export_tab3:
        st.subheader("Admission Letters")
        letters_data = create_admission_letters_data(admission_results, original_df)
        if not letters_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Letter Template Data**")
                st.dataframe(letters_data[['name', 'admitted_course', 'letter_type']].head(), use_container_width=True)
            with col2:
                st.markdown("**Download Options**")
                letters_btn = create_download_button(
                    letters_data,
                    f"admission_letters_data_{timestamp}.csv",
                    "📥 Download Letters Data"
                )
                if letters_btn:
                    st.markdown(letters_btn, unsafe_allow_html=True)
                if st.button("📜 Generate PDF Letters"):
                    with st.spinner("Generating PDF letters..."):
                        for _, student in letters_data.iterrows():
                            pdf_buffer = generate_admission_letter(student)
                            pdf_btn = create_download_button(
                                pdf_buffer,
                                f"admission_letter_{student['student_id']}_{timestamp}.pdf",
                                f"📥 Letter for {student['name']}",
                                "pdf"
                            )
                            if pdf_btn:
                                st.markdown(pdf_btn, unsafe_allow_html=True)
        else:
            st.info("No admitted students to generate letters for.")
    
    with export_tab4:
        st.subheader("Statistical Analysis Report")
        stats_report = generate_statistical_report(admission_results, original_df, course_capacities)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Application Summary**")
            for key, value in stats_report['application_summary'].items():
                st.metric(key.replace('_', ' ').title(), value)
        with col2:
            st.markdown("**Score Analysis**")
            for key, value in stats_report['score_analysis'].items():
                if value != "N/A":
                    st.metric(key.replace('_', ' ').title(), value)
        stats_json_btn = create_download_button(
            stats_report,
            f"statistical_analysis_{timestamp}.json",
            "📥 Download Statistical Report",
            "json"
        )
        if stats_json_btn:
            st.markdown(stats_json_btn, unsafe_allow_html=True)
    
    with demand_tab:
        st.subheader("Student Demand Analysis")
        demand_df = pd.DataFrame.from_dict(demand_analysis, orient='index')
        st.dataframe(demand_df, use_container_width=True)
        fig = px.bar(demand_df, x=demand_df.index, y='total_estimated_demand', title="Estimated Course Demand")
        st.plotly_chart(fig, use_container_width=True)
    
    with capacity_tab:
        st.subheader("Capacity Optimization Suggestions")
        if capacity_suggestions:
            suggestions_df = pd.DataFrame.from_dict(capacity_suggestions, orient='index')
            st.dataframe(suggestions_df, use_container_width=True)
            fig = px.bar(suggestions_df, x=suggestions_df.index, y='suggested', title="Suggested Capacity Adjustments")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant optimizations suggested.")

def export_to_excel(admission_results, original_df, course_capacities, filename):
    """Export results to Excel"""
    detailed_results, summary_stats, course_breakdown = create_comprehensive_admission_report(admission_results, original_df, course_capacities)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        detailed_results.to_excel(writer, sheet_name='All_Results', index=False)
        worksheet1 = writer.sheets['All_Results']
        for col_num, value in enumerate(detailed_results.columns.values):
            worksheet1.write(0, col_num, value, header_format)
        admitted_students = detailed_results[detailed_results['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]
        admitted_students.to_excel(writer, sheet_name='Admitted_Students', index=False)
        worksheet2 = writer.sheets['Admitted_Students']
        for col_num, value in enumerate(admitted_students.columns.values):
            worksheet2.write(0, col_num, value, header_format)
        course_breakdown.to_excel(writer, sheet_name='Course_Analysis', index=False)
        worksheet3 = writer.sheets['Course_Analysis']
        for col_num, value in enumerate(course_breakdown.columns.values):
            worksheet3.write(0, col_num, value, header_format)
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        worksheet4 = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet4.write(0, col_num, value, header_format)
    output.seek(0)
    return output

def create_admission_letters_data(admission_results, original_df):
    """Create data for admission letters"""
    results_df = pd.DataFrame(admission_results)
    admitted_students = results_df[results_df['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]
    letters_data = admitted_students.merge(
        original_df[['student_id', 'name', 'utme_score', 'preferred_course']], 
        on='student_id', 
        how='left'
    )
    letters_data['admission_date'] = datetime.now().strftime('%B %d, %Y')
    letters_data['academic_session'] = f"{datetime.now().year}/{datetime.now().year + 1}"
    letters_data['letter_type'] = letters_data['status'].map({
        'ADMITTED': 'Direct Admission',
        'ALTERNATIVE_ADMISSION': 'Alternative Course Admission'
    })
    return letters_data

def generate_statistical_report(admission_results, original_df, course_capacities):
    """Generate statistical report"""
    results_df = pd.DataFrame(admission_results)
    total_applications = len(original_df)
    admitted_count = len(results_df[results_df['status'] == 'ADMITTED'])
    alternative_count = len(results_df[results_df['status'] == 'ALTERNATIVE_ADMISSION'])
    rejected_count = len(results_df[results_df['status'] == 'NOT_QUALIFIED'])
    waitlisted_count = len(results_df[results_df['status'] == 'WAITLISTED'])
    admitted_scores = results_df[results_df['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]['score']
    preference_analysis = original_df['preferred_course'].value_counts()
    utme_stats = original_df['utme_score'].describe()
    utilization_stats = calculate_capacity_utilization(admission_results, course_capacities)
    
    report = {
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'application_summary': {
            'total_applications': total_applications,
            'admitted_direct': admitted_count,
            'admitted_alternative': alternative_count,
            'rejected': rejected_count,
            'waitlisted': waitlisted_count,
            'admission_rate': f"{((admitted_count + alternative_count) / total_applications * 100):.1f}%"
        },
        'score_analysis': {
            'mean_admitted_score': f"{admitted_scores.mean():.3f}" if not admitted_scores.empty else "N/A",
            'median_admitted_score': f"{admitted_scores.median():.3f}" if not admitted_scores.empty else "N/A",
            'min_admitted_score': f"{admitted_scores.min():.3f}" if not admitted_scores.empty else "N/A",
            'max_admitted_score': f"{admitted_scores.max():.3f}" if not admitted_scores.empty else "N/A"
        },
        'utme_distribution': {
            'mean': f"{utme_stats['mean']:.1f}",
            'median': f"{utme_stats['50%']:.1f}",
            'min': f"{utme_stats['min']:.0f}",
            'max': f"{utme_stats['max']:.0f}",
            'std': f"{utme_stats['std']:.1f}"
        },
        'top_preferences': preference_analysis.head(10).to_dict(),
        'capacity_utilization': {
            course: {
                'capacity': stats['capacity'],
                'admitted': stats['admitted'],
                'utilization_rate': f"{stats['utilization_rate']:.1f}%"
            }
            for course, stats in utilization_stats.items()
            if stats['admitted'] > 0
        }
    }
    return report

# Main application
st.header("📁 FUTA Admission Processing System")
uploaded_file = st.file_uploader("Upload CSV with student data (student_id, name, utme_score, preferred_course, interests, learning_style, state_of_origin, gender, age, [subject]_grade)", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        course_capacities = get_dynamic_course_capacities(df)
        with st.spinner("Processing admissions (this may take a minute)..."):
            admission_results = process_csv_applications(df, course_capacities)
        display_comprehensive_dashboard(admission_results, df, course_capacities)
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        logger.error(f"Error in main application: {str(e)}")

# Individual recommendation section
st.markdown("---")
st.header("🎓 Individual Student Recommendation")
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

st.sidebar.header("📝 Enter Your Details")
utme_score = st.sidebar.number_input(
    "UTME Score (0-400)",
    min_value=0,
    max_value=400,
    value=250,
    step=1,
    help="Enter your Unified Tertiary Matriculation Examination score"
)
st.sidebar.subheader("🎯 Your Interests")
selected_interests = st.sidebar.multiselect(
    "Select your areas of interest (choose 2-4):",
    options=list(interest_categories.keys()),
    default=["Problem Solving & Logic"],
    help="Select multiple Fence:areas that interest you most"
)
learning_style = st.sidebar.selectbox(
    "Learning Style",
    options=list(learning_styles.keys()),
    help="How do you prefer to learn and work?"
)
state_of_origin = st.sidebar.selectbox(
    "State of Origin",
    options=NIGERIAN_STATES,
    help="Select your state of origin"
)
gender = st.sidebar.selectbox(
    "Gender",
    options=["Male", "Female", "Other"],
    help="Select your gender"
)
st.sidebar.subheader("📚 O'Level Results")
olevel_subjects = {}
for i in range(7):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        subject = st.selectbox(
            f"Subject {i+1}",
            options=[""] + common_subjects,
            key=f"subject_{i}",
            label_visibility="collapsed" if i > 0 else "visible"
        )
    with col2:
        if subject:
            grade = st.selectbox(
                f"Grade {i+1}",
                options=list(grade_map.keys()),
                key=f"grade_{i}",
                label_visibility="collapsed"
            )
            olevel_subjects[subject] = grade_map[grade]

if st.sidebar.button("🔮 Get My Recommendations", type="primary"):
    try:
        if len(olevel_subjects) >= 5 and len(selected_interests) >= 1:
            st.session_state.prediction_made = True
            with st.spinner("Generating recommendations (please wait)..."):
                st.session_state.prediction_result = predict_placement_enhanced(
                    utme_score, olevel_subjects, selected_interests, learning_style, state_of_origin, gender
                )
                st.session_state.user_data = {
                    "utme_score": utme_score,
                    "olevel_subjects": olevel_subjects,
                    "selected_interests": selected_interests,
                    "learning_style": learning_style,
                    "state_of_origin": state_of_origin,
                    "gender": gender
                }
        else:
            st.sidebar.error("Please enter at least 5 O'Level subjects and select at least 1 interest area")
    except Exception as e:
        st.sidebar.error(f"Recommendation failed: {str(e)}")
        logger.error(f"Error in recommendation button: {str(e)}")

if st.session_state.prediction_made:
    result = st.session_state.prediction_result
    user_data = st.session_state.user_data

    st.header("🎯 Your Personalized Course Recommendations")
    if result["predicted_program"] != "UNASSIGNED":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="🏆 Top Recommendation",
                value=result["predicted_program"],
                help="Your best course match based on comprehensive analysis"
            )
        with col2:
            st.metric(
                label="📊 Match Score",
                value=f"{result['score']:.1%}",
                help="Overall compatibility score"
            )
        with col3:
            st.metric(
                label="🎯 Interest Alignment",
                value="✅ High" if result.get("interest_alignment", False) else "⚠️ Moderate"
            )
        with col4:
            st.metric(
                label="📈 UTME Status",
                value=f"{user_data['utme_score']}/{cutoff_marks[result['predicted_program']]}",
                delta=f"{user_data['utme_score'] - cutoff_marks[result['predicted_program']]} above cutoff"
            )
        
        if result["predicted_program"] in course_details:
            st.subheader("📋 Course Information")
            course_info = course_details[result["predicted_program"]]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Description:** {course_info['description']}")
                st.markdown(f"**Duration:** {course_info['duration']}")
                st.markdown(f"**Job Outlook:** {course_info['job_outlook']}")
            with col2:
                st.markdown(f"**Average Salary:** {course_info['average_salary']}")
                st.markdown("**Career Paths:**")
                for path in course_info['career_paths']:
                    st.markdown(f"• {path}")
        
        if "all_eligible" in result and len(result["all_eligible"]) > 1:
            st.subheader("📊 All Eligible Courses Analysis")
            eligible_df = result["all_eligible"].head(10)
            fig = go.Figure()
            colors = ['#1f77b4' if course == result["predicted_program"] else '#aec7e8' 
                     for course in eligible_df["course"]]
            fig.add_trace(go.Bar(
                y=eligible_df["course"],
                x=eligible_df["score"],
                orientation='h',
                marker_color=colors,
                text=[f"{score:.1%}" for score in eligible_df["score"]],
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 10 Course Matches",
                xaxis_title="Match Score",
                yaxis_title="Course",
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            display_df = eligible_df[["course", "score"]].rename(columns={"course": "Course", "score": "Match Score"})
            st.dataframe(display_df, use_container_width=True)
    else:
        st.error("❌ No Eligible Course Found")
        st.markdown(f"**Reason:** {result['reason']}")
        st.subheader("💡 Improvement Suggestions")
        for suggestion in result.get("suggestions", []):
            st.markdown(f"• {suggestion}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <h4>🎓 FUTA Admission Management System</h4>
        <p>Powered by advanced ML for intelligent course allocation</p>
        <p><em>Optimizing admissions for fairness and efficiency</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
