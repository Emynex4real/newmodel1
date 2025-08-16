# Consolidated Imports (covering all parts)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import logging
import traceback
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
from datetime import datetime
import plotly.express as px
import asyncio

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Part 1: Constants and Helper Functions ---
# (Contents of app_part1.py, artifact ID: 0b748fd6-88ca-4ad8-ab07-7a4bb23bbea5)
# Constants
NIGERIAN_STATES = ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara", "FCT"]
common_subjects = ["English Language", "Mathematics", "Physics", "Chemistry", "Biology", "Economics"]
grade_map = {"A1": 6, "B2": 5, "B3": 4, "C4": 3, "C5": 2, "C6": 1}
learning_styles = {
    "Analytical Thinker": ["Computer Science", "Mechanical Engineering"],
    "Visual Learner": ["Physiology"],
    "Practical Learner": ["Mechanical Engineering"]
}
interest_categories = {
    "Problem Solving & Logic": ["Computer Science"],
    "Technology & Innovation": ["Computer Science", "Mechanical Engineering"],
    "Healthcare & Medicine": ["Physiology"]
}
course_names = ["Computer Science", "Physiology", "Mechanical Engineering"]
cutoff_marks = {
    "Computer Science": 200,
    "Physiology": 180,
    "Mechanical Engineering": 190
}

# Helper functions (replace with your actual implementations from app_part1.py)
def get_course_requirements():
    return {
        "Computer Science": {
            "olevel_required": ["English Language", "Mathematics", "Physics"],
            "olevel_any": ["Chemistry", "Biology", "Economics"],
            "min_grades": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6"},
            "utme_required": ["English Language", "Mathematics", "Physics"],
            "utme_any": ["Chemistry", "Biology"]
        },
        "Physiology": {
            "olevel_required": ["English Language", "Mathematics", "Biology", "Chemistry"],
            "olevel_any": ["Physics", "Economics"],
            "min_grades": {"English Language": "C6", "Mathematics": "C6", "Biology": "C6", "Chemistry": "C6"},
            "utme_required": ["English Language", "Biology", "Chemistry"],
            "utme_any": ["Physics"]
        },
        "Mechanical Engineering": {
            "olevel_required": ["English Language", "Mathematics", "Physics", "Chemistry"],
            "olevel_any": ["Biology", "Economics"],
            "min_grades": {"English Language": "C6", "Mathematics": "C6", "Physics": "C6", "Chemistry": "C6"},
            "utme_required": ["English Language", "Mathematics", "Physics"],
            "utme_any": ["Chemistry"]
        }
    }

def is_eligible(olevel_subjects, utme_subjects, course, parsed_requirements):
    req = parsed_requirements.get(course, {})
    required_olevel = req.get("olevel_required", [])
    any_olevel = req.get("olevel_any", [])
    min_grades = req.get("min_grades", {})
    required_utme = req.get("utme_required", [])
    any_utme = req.get("utme_any", [])
    
    olevel_valid = all(sub in olevel_subjects for sub in required_olevel)
    olevel_any_valid = len([sub for sub in olevel_subjects if sub in any_olevel]) >= (5 - len(required_olevel))
    grade_valid = all(olevel_subjects.get(sub, 0) >= grade_map.get(min_grades.get(sub, "C6"), 1) for sub in required_olevel)
    utme_valid = all(sub in utme_subjects for sub in required_utme)
    utme_any_valid = len([sub for sub in utme_subjects if sub in any_utme]) >= (4 - len(required_utme))
    
    return olevel_valid and olevel_any_valid and grade_valid and utme_valid and utme_any_valid

def compute_grade_sum(olevel_subjects, course, parsed_requirements):
    req = parsed_requirements.get(course, {})
    required = req.get("olevel_required", [])
    any_subjects = req.get("olevel_any", [])
    grades = [olevel_subjects.get(sub, 0) for sub in required]
    any_grades = [olevel_subjects.get(sub, 0) for sub in any_subjects if sub in olevel_subjects]
    grades.extend(any_grades[:max(0, 5 - len(required))])
    return sum(grades), len(grades)

def compute_enhanced_score(utme_score, grade_sum, count, interest_weight, diversity_score):
    utme_weight = utme_score / 400 * 0.5
    olevel_weight = grade_sum / (count * 6) * 0.3 if count > 0 else 0
    return min(utme_weight + olevel_weight + interest_weight + diversity_score, 1.0)

def get_dynamic_course_capacities(df):
    return {course: 100 for course in course_names}  # Placeholder; replace with actual logic

def create_comprehensive_admission_report(admission_results, df, course_capacities):
    detailed_results = pd.DataFrame(admission_results)
    summary_stats = {
        'Total Applicants': len(admission_results),
        'Admitted Students': sum(1 for r in admission_results if r['status'] in ['ADMITTED', 'ALTERNATIVE_ADMISSION'])
    }
    course_breakdown = pd.DataFrame([
        {
            'admitted_course': course,
            'Students_Admitted': sum(1 for r in admission_results if r['admitted_course'] == course),
            'Avg_Score': np.mean([r['score'] for r in admission_results if r['admitted_course'] == course]) if any(r['admitted_course'] == course for r in admission_results) else 0,
            'Min_Score': min([r['score'] for r in admission_results if r['admitted_course'] == course], default=0),
            'Max_Score': max([r['score'] for r in admission_results if r['admitted_course'] == course], default=0),
            'Capacity': course_capacities.get(course, 0),
            'Utilization_Rate': (sum(1 for r in admission_results if r['admitted_course'] == course) / course_capacities.get(course, 1)) * 100
        } for course in course_names
    ])
    return detailed_results, summary_stats, course_breakdown

def generate_admission_letter(student_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, f"Admission Letter for {student_data['name']}")
    c.drawString(100, 730, f"Student ID: {student_data['student_id']}")
    c.drawString(100, 710, f"Admitted to: {student_data['admitted_course']}")
    c.save()
    buffer.seek(0)
    return buffer

def analyze_student_demand(df):
    return {
        course: {
            'primary_demand': sum(1 for _, row in df.iterrows() if row['preferred_course'] == course),
            'secondary_demand': 0,
            'total_estimated_demand': sum(1 for _, row in df.iterrows() if row['preferred_course'] == course),
            'demand_category': 'High' if sum(1 for _, row in df.iterrows() if row['preferred_course'] == course) > 50 else 'Low'
        } for course in course_names
    }

def optimize_course_capacities(demand, capacities):
    return {
        course: {
            'current_capacity': capacities.get(course, 0),
            'suggested_capacity': max(capacities.get(course, 0), demand.get(course, 0)),
            'reason': 'Based on demand',
            'priority': 'High' if demand.get(course, 0) > capacities.get(course, 0) else 'Low'
        } for course in course_names
    }

def calculate_capacity_utilization(admission_results, course_capacities):
    return {
        course: {
            'Capacity': course_capacities.get(course, 0),
            'Admitted': sum(1 for r in admission_results if r['admitted_course'] == course),
            'Available': course_capacities.get(course, 0) - sum(1 for r in admission_results if r['admitted_course'] == course),
            'Utilization Rate': (sum(1 for r in admission_results if r['admitted_course'] == course) / course_capacities.get(course, 1)) * 100,
            'Status': 'Optimal' if (sum(1 for r in admission_results if r['admitted_course'] == course) / course_capacities.get(course, 1)) < 0.9 else 'Overcapacity'
        } for course in course_names
    }

# --- END OF PART 1 ---
# Replace any placeholder predict_placement_enhanced or run_intelligent_admission_algorithm_v2 with the ML versions below

# --- Part 2: ML Models and Core Processing ---
# (Contents of app_part2_ml.py, artifact ID: e5ee0ae7-5634-4ebf-b286-f6b9b8bb2801)
ELIGIBILITY_MODEL = None
ELIGIBILITY_SCALER = None
ELIGIBILITY_FEATURES = None
SCORING_MODEL = None
SCORING_SCALER = None
SCORING_FEATURES = None

def train_eligibility_model():
    global ELIGIBILITY_MODEL, ELIGIBILITY_SCALER, ELIGIBILITY_FEATURES
    logger.info("Training eligibility model")
    data = []
    parsed_req = get_course_requirements()
    for _ in range(500):
        utme = np.random.randint(100, 400)
        num_subjects = np.random.randint(5, len(common_subjects) + 1)
        selected_olevel_subs = np.random.choice(common_subjects, num_subjects, replace=False)
        olevels = {sub: np.random.choice(list(grade_map.values())) for sub in selected_olevel_subs}
        utme_subjects = ["English Language"] + list(np.random.choice(
            [s for s in common_subjects if s != "English Language"], 3, replace=False))
        interests = np.random.choice(list(interest_categories.keys()), np.random.randint(1, len(interest_categories.keys()) + 1), replace=False).tolist()
        learning = np.random.choice(list(learning_styles.keys()))
        state = np.random.choice(NIGERIAN_STATES)
        gender = np.random.choice(["Male", "Female", "Other"])
        
        for course in course_names:
            eligible = is_eligible(olevels, utme_subjects, course, parsed_req) and utme >= cutoff_marks.get(course, 180)
            features = {
                'utme': utme,
                'eligible': 1 if eligible else 0
            }
            for sub in common_subjects:
                features[sub] = olevels.get(sub, 9)
            for utme_sub in common_subjects:
                features[f'utme_{utme_sub}'] = 1 if utme_sub in utme_subjects else 0
            for int_ in interest_categories.keys():
                features[int_] = 1 if int_ in interests else 0
            for ls in learning_styles.keys():
                features[f'ls_{ls}'] = 1 if learning == ls else 0
            features['diversity_score'] = 0.5 if state in ["Yobe", "Zamfara", "Borno"] else 0.3 if gender == "Female" else 0
            features['course'] = course
            data.append(features)

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['course'])
    X = df.drop('eligible', axis=1)
    y = df['eligible']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Eligibility model trained with accuracy: {accuracy:.4f}")
    ELIGIBILITY_MODEL = model
    ELIGIBILITY_SCALER = scaler
    ELIGIBILITY_FEATURES = X.columns.tolist()

def train_scoring_model():
    global SCORING_MODEL, SCORING_SCALER, SCORING_FEATURES
    logger.info("Training scoring model")
    data = []
    parsed_req = get_course_requirements()
    for _ in range(500):
        utme = np.random.randint(100, 400)
        num_subjects = np.random.randint(5, len(common_subjects) + 1)
        selected_olevel_subs = np.random.choice(common_subjects, num_subjects, replace=False)
        olevels = {sub: np.random.choice(list(grade_map.values())) for sub in selected_olevel_subs}
        utme_subjects = ["English Language"] + list(np.random.choice(
            [s for s in common_subjects if s != "English Language"], 3, replace=False))
        interests = np.random.choice(list(interest_categories.keys()), np.random.randint(1, len(interest_categories.keys()) + 1), replace=False).tolist()
        learning = np.random.choice(list(learning_styles.keys()))
        state = np.random.choice(NIGERIAN_STATES)
        gender = np.random.choice(["Male", "Female", "Other"])
        
        for course in course_names:
            eligible = is_eligible(olevels, utme_subjects, course, parsed_req) and utme >= cutoff_marks.get(course, 180)
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
                'score': score
            }
            for sub in common_subjects:
                features[sub] = olevels.get(sub, 9)
            for utme_sub in common_subjects:
                features[f'utme_{utme_sub}'] = 1 if utme_sub in utme_subjects else 0
            for int_ in interest_categories.keys():
                features[int_] = 1 if int_ in interests else 0
            for ls in learning_styles.keys():
                features[f'ls_{ls}'] = 1 if learning == ls else 0
            features['diversity_score'] = diversity_score
            features['course'] = course
            data.append(features)

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['course'])
    X = df.drop('score', axis=1)
    y = df['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Scoring model trained with MSE: {mse:.4f}")
    SCORING_MODEL = model
    SCORING_SCALER = scaler
    SCORING_FEATURES = X.columns.tolist()

# Initialize models
train_eligibility_model()
train_scoring_model()

def predict_placement_enhanced(utme_score, olevel_subjects, utme_subjects, selected_interests, learning_style, state, gender):
    global ELIGIBILITY_MODEL, ELIGIBILITY_SCALER, ELIGIBILITY_FEATURES, SCORING_MODEL, SCORING_SCALER, SCORING_FEATURES
    results = []
    diversity_score = 0.5 if state in ["Yobe", "Zamfara", "Borno"] else 0.3 if gender == "Female" else 0
    for course in course_names:
        features = {
            'utme': utme_score,
        }
        for sub in common_subjects:
            features[sub] = olevel_subjects.get(sub, 9)
        for utme_sub in common_subjects:
            features[f'utme_{utme_sub}'] = 1 if utme_sub in utme_subjects else 0
        for int_ in interest_categories.keys():
            features[int_] = 1 if int_ in selected_interests else 0
        for ls in learning_styles.keys():
            features[f'ls_{ls}'] = 1 if learning_style == ls else 0
        features['diversity_score'] = diversity_score
        for c in course_names:
            features[f'course_{c}'] = 1 if c == course else 0

        # Predict eligibility using ML model
        features_df = pd.DataFrame([features])
        for feature in ELIGIBILITY_FEATURES:
            if feature not in features_df.columns:
                features_df[feature] = 0
        features_df = features_df[ELIGIBILITY_FEATURES]
        X_scaled = ELIGIBILITY_SCALER.transform(features_df)
        eligible = ELIGIBILITY_MODEL.predict(X_scaled)[0]

        # Predict score using ML model
        features_df = pd.DataFrame([features])
        for feature in SCORING_FEATURES:
            if feature not in features_df.columns:
                features_df[feature] = 0
        features_df = features_df[SCORING_FEATURES]
        X_scaled = SCORING_SCALER.transform(features_df)
        score = SCORING_MODEL.predict(X_scaled)[0]

        interest_weight = sum(1 for int_ in selected_interests if course in interest_categories.get(int_, [])) * 0.3
        if learning_style in learning_styles and course in learning_styles[learning_style]:
            interest_weight += 0.2

        results.append({
            "course": course,
            "eligible": eligible == 1,
            "score": score,
            "interest_weight": interest_weight,
            "diversity_score": diversity_score
        })

    results_df = pd.DataFrame(results)
    eligible_courses = results_df[results_df["eligible"] & (results_df["score"] > 0)]
    if eligible_courses.empty:
        return {
            "predicted_program": "UNASSIGNED",
            "score": 0,
            "reason": "No eligible courses based on ML prediction",
            "all_eligible": pd.DataFrame(),
            "suggestions": ["Improve UTME score", "Add more relevant O'Level subjects"]
        }
    best_course = eligible_courses.loc[eligible_courses["score"].idxmax()]
    return {
        "predicted_program": best_course["course"],
        "score": best_course["score"],
        "reason": "Best match based on ML prediction",
        "all_eligible": eligible_courses.sort_values("score", ascending=False),
        "interest_alignment": best_course["interest_weight"] > 0
    }

def run_intelligent_admission_algorithm_v2(students, course_capacities):
    admission_results = []
    for student in students:
        prediction = predict_placement_enhanced(
            student['utme_score'],
            student['olevel_subjects'],
            student['utme_subjects'],
            student['interests'],
            student['learning_style'],
            student['state_of_origin'],
            student['gender']
        )
        status = "ADMITTED" if prediction['predicted_program'] != "UNASSIGNED" else "NOT_ADMITTED"
        admitted_course = prediction['predicted_program']
        admission_type = "Merit" if status == "ADMITTED" and student['preferred_course'] == admitted_course else "Alternative" if status == "ADMITTED" else "None"
        admission_results.append({
            'student_id': student['student_id'],
            'name': student['name'],
            'admitted_course': admitted_course,
            'status': status,
            'admission_type': admission_type,
            'score': prediction['score'],
            'rank': 1,
            'reason': prediction['reason'],
            'original_preference': student['preferred_course'],
            'recommendation_reason': "ML-based prediction" if status == "ADMITTED" else "",
            'available_alternatives': len(prediction['all_eligible']) - 1 if not prediction['all_eligible'].empty else 0,
            'suggested_alternatives': prediction['all_eligible']['course'].tolist()[1:] if len(prediction['all_eligible']) > 1 else []
        })
    return admission_results

async def process_csv_applications(df, course_capacities, progress_callback=None):
    logger.info("Starting CSV processing: %s rows", len(df))
    required_columns = [
        'student_id', 'name', 'utme_score', 'preferred_course', 'utme_subjects',
        'interests', 'learning_style', 'state_of_origin', 'gender',
        'english_language_grade', 'mathematics_grade'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error("Missing required columns in CSV: %s", missing_columns)
        return [], pd.DataFrame(), [{'row': 0, 'student_id': 'N/A', 'error': f"Missing columns: {', '.join(missing_columns)}"}]

    students = []
    eligible_courses_list = []
    invalid_rows = []
    total_rows = len(df)

    for index, row in df.iterrows():
        try:
            logger.debug("Processing row %s, student_id: %s", index + 2, row.get('student_id', 'Unknown'))
            utme_subjects = (
                [s.strip() for s in row['utme_subjects'].split(',')]
                if pd.notna(row['utme_subjects']) and row['utme_subjects'].strip()
                else []
            )
            if len(utme_subjects) != 4 or "English Language" not in utme_subjects:
                logger.warning("Invalid UTME subjects for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, utme_subjects)
                invalid_rows.append({'row': index + 2, 'student_id': row.get('student_id', 'Unknown'), 'error': 'Invalid UTME subjects'})
                continue

            interests = (
                [i.strip() for i in row['interests'].split(',')]
                if pd.notna(row['interests']) and row['interests'].strip()
                else []
            )

            olevel_subjects = {}
            for subject in ['english_language_grade', 'mathematics_grade', 'physics_grade',
                          'chemistry_grade', 'biology_grade', 'economics_grade']:
                if subject in row and pd.notna(row[subject]) and row[subject] in grade_map:
                    subject_name = subject.replace('_grade', '').replace('_', ' ').title()
                    olevel_subjects[subject_name] = grade_map[row[subject]]
            if len(olevel_subjects) < 5 or 'English Language' not in olevel_subjects or 'Mathematics' not in olevel_subjects:
                logger.warning("Insufficient or invalid O'Level subjects for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, olevel_subjects)
                invalid_rows.append({'row': index + 2, 'student_id': row.get('student_id', 'Unknown'), 'error': 'Invalid O\'Level subjects'})
                continue

            if pd.isna(row['student_id']) or not row['student_id']:
                logger.warning("Missing student_id at row %s", index + 2)
                invalid_rows.append({'row': index + 2, 'student_id': 'Unknown', 'error': 'Missing student_id'})
                continue
            if pd.isna(row['name']) or not row['name'].strip():
                logger.warning("Missing name for student %s at row %s", row['student_id'], index + 2)
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Missing name'})
                continue
            if pd.isna(row['utme_score']) or row['utme_score'] <= 0:
                logger.warning("Invalid UTME score for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, row['utme_score'])
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Invalid UTME score'})
                continue
            if pd.isna(row['preferred_course']) or row['preferred_course'] not in course_names:
                logger.warning("Invalid preferred course for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, row['preferred_course'])
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Invalid preferred course'})
                continue
            if pd.isna(row['learning_style']) or row['learning_style'] not in learning_styles:
                logger.warning("Invalid learning style for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, row['learning_style'])
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Invalid learning style'})
                continue
            if pd.isna(row['state_of_origin']) or row['state_of_origin'] not in NIGERIAN_STATES:
                logger.warning("Invalid state of origin for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, row['state_of_origin'])
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Invalid state of origin'})
                continue
            if pd.isna(row['gender']) or row['gender'] not in ['Male', 'Female', 'Other']:
                logger.warning("Invalid gender for student %s at row %s: %s",
                              row.get('student_id', 'Unknown'), index + 2, row['gender'])
                invalid_rows.append({'row': index + 2, 'student_id': row['student_id'], 'error': 'Invalid gender'})
                continue

            student = {
                'student_id': str(row['student_id']),
                'name': row['name'].strip(),
                'utme_score': float(row['utme_score']),
                'preferred_course': row['preferred_course'].strip(),
                'utme_subjects': utme_subjects,
                'interests': interests,
                'learning_style': row['learning_style'].strip(),
                'state_of_origin': row['state_of_origin'].strip(),
                'gender': row['gender'].strip(),
                'olevel_subjects': olevel_subjects
            }
            students.append(student)

            prediction = predict_placement_enhanced(
                student['utme_score'],
                student['olevel_subjects'],
                student['utme_subjects'],
                student['interests'],
                student['learning_style'],
                student['state_of_origin'],
                student['gender']
            )
            if 'all_eligible' in prediction and not prediction['all_eligible'].empty:
                eligible_df = prediction['all_eligible'][['course', 'score', 'interest_weight', 'diversity_score']].copy()
                eligible_df['student_id'] = student['student_id']
                eligible_df['name'] = student['name']
                eligible_courses_list.append(eligible_df)
            else:
                eligible_courses_list.append(pd.DataFrame([{
                    'student_id': student['student_id'],
                    'name': student['name'],
                    'course': 'NONE',
                    'score': 0.0,
                    'interest_weight': 0.0,
                    'diversity_score': 0.0
                }]))
            logger.debug("Completed processing for student %s", student['student_id'])

            if progress_callback:
                progress_callback((index + 1) / total_rows)

        except Exception as e:
            logger.error("Error processing row %s for student %s: %s", index + 2, row.get('student_id', 'Unknown'), str(e))
            logger.error(traceback.format_exc())
            invalid_rows.append({'row': index + 2, 'student_id': row.get('student_id', 'Unknown'), 'error': str(e)})
            continue

    if invalid_rows:
        logger.info("Invalid rows detected: %s", len(invalid_rows))

    if not students:
        logger.error("No valid student records found in CSV after processing")
        return [], pd.DataFrame(), invalid_rows

    try:
        logger.info("Running admission algorithm for %s students", len(students))
        admission_results = run_intelligent_admission_algorithm_v2(students, course_capacities)
        eligible_courses_df = pd.concat(eligible_courses_list, ignore_index=True) if eligible_courses_list else pd.DataFrame()
        logger.info("Batch processing completed: %s results generated, %s eligible courses records",
                    len(admission_results), len(eligible_courses_df))
        return admission_results, eligible_courses_df, invalid_rows
    except Exception as e:
        logger.error("Error in batch admission processing: %s", str(e))
        logger.error(traceback.format_exc())
        return [], pd.DataFrame(), invalid_rows

# --- Part 3: UI and Main Function ---
# (Contents of app_part3_ml.py, artifact ID: 22469383-6bde-46df-a129-e60357e6041d)
async def process_with_timeout(coro, timeout=300):
    """Run an async coroutine with a timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Processing timed out after %s seconds", timeout)
        st.error(f"Processing timed out after {timeout} seconds. Try uploading a smaller CSV or check the logs.")
        return [], pd.DataFrame(), [{'row': 0, 'student_id': 'N/A', 'error': 'Processing timed out'}]

def main():
    """Main Streamlit app"""
    st.title("🎓 FUTA Intelligent Admission Management System (ML-Based)")
    st.markdown("Welcome to the Federal University of Technology, Akure Admission Management System. This system uses machine learning models to predict eligibility and course placement scores.")

    # Initialize session state for form key and uploader key
    if 'form_key_counter' not in st.session_state:
        st.session_state.form_key_counter = 0
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    tab1, tab2, tab3 = st.tabs(["Individual Admission", "Batch Processing", "Analytics & Insights"])

    with tab1:
        st.header("Individual Admission Prediction")
        form_key = f"individual_prediction_form_{st.session_state.form_key_counter}"

        with st.form(form_key, clear_on_submit=True):
            st.subheader("Candidate Information")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", placeholder="Enter your full name", key=f"name_{st.session_state.form_key_counter}")
                state = st.selectbox("State of Origin", NIGERIAN_STATES, index=28, key=f"state_{st.session_state.form_key_counter}")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key=f"gender_{st.session_state.form_key_counter}")
            with col2:
                utme_score = st.number_input("UTME Score", min_value=0, max_value=400, step=1, key=f"utme_score_{st.session_state.form_key_counter}")
                learning_style = st.selectbox("Learning Style", list(learning_styles.keys()), index=2, key=f"learning_style_{st.session_state.form_key_counter}")

            st.subheader("UTME Subjects")
            utme_subjects = st.multiselect(
                "Select 4 UTME Subjects (English Language is mandatory)",
                common_subjects,
                default=["English Language"],
                max_selections=4,
                key=f"utme_subjects_{st.session_state.form_key_counter}"
            )
            utme_subjects = utme_subjects if utme_subjects is not None else []

            st.subheader("O'Level Results")
            st.write("Select at least 5 subjects and their grades")
            olevel_subjects = {}
            for i in range(5):
                col1, col2 = st.columns(2)
                with col1:
                    subject = st.selectbox(f"O'Level Subject {i+1}", [""] + common_subjects, key=f"olevel_sub_{i}_{st.session_state.form_key_counter}")
                with col2:
                    grade = st.selectbox(f"Grade {i+1}", list(grade_map.keys()), key=f"olevel_grade_{i}_{st.session_state.form_key_counter}")
                if subject and subject != "":
                    olevel_subjects[subject] = grade_map[grade]

            st.subheader("Interests")
            interests = st.multiselect("Select Your Interests", list(interest_categories.keys()), key=f"interests_{st.session_state.form_key_counter}")
            interests = interests if interests is not None else []

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
                elif not olevel_subjects or len(olevel_subjects) < 5:
                    st.error("Please select at least 5 O'Level subjects with valid grades.")
                elif "English Language" not in utme_subjects:
                    st.error("English Language is a mandatory UTME subject.")
                else:
                    try:
                        prediction = predict_placement_enhanced(
                            utme_score,
                            olevel_subjects,
                            utme_subjects,
                            interests,
                            learning_style,
                            state,
                            gender
                        )
                        if prediction['predicted_program'] == "UNASSIGNED":
                            st.error(prediction['reason'])
                            if 'suggestions' in prediction:
                                st.subheader("Suggestions for Improvement")
                                for suggestion in prediction['suggestions']:
                                    st.markdown(f"- {suggestion}")
                        else:
                            st.success(f"Congratulations! You are predicted to be admitted to **{prediction['predicted_program']}**")
                            st.write(f"**Prediction Score**: {prediction['score']:.2f}")
                            st.write(f"**Reason**: {prediction['reason']}")
                            if prediction['interest_alignment']:
                                st.write("✅ This course aligns well with your interests!")
                            if not prediction['all_eligible'].empty:
                                st.subheader("Other Eligible Courses")
                                st.dataframe(
                                    prediction['all_eligible'][['course', 'score', 'interest_weight', 'diversity_score']],
                                    column_config={
                                        "course": "Course",
                                        "score": st.column_config.NumberColumn("Prediction Score", format="%.2f"),
                                        "interest_weight": st.column_config.NumberColumn("Interest Alignment", format="%.2f"),
                                        "diversity_score": st.column_config.NumberColumn("Diversity Score", format="%.2f")
                                    },
                                    hide_index=True
                                )
                        st.session_state.form_key_counter += 1
                    except Exception as e:
                        logger.error(f"Error during individual prediction: {str(e)}")
                        logger.error(traceback.format_exc())
                        st.error(f"Prediction failed: {str(e)}")

    with tab2:
        st.header("Batch Admission Processing")
        st.write("Upload a CSV file with student data to process admissions in bulk.")
        st.markdown("""
            **Expected CSV Format**:
            - Columns: `student_id`, `name`, `utme_score`, `preferred_course`, `utme_subjects` (comma-separated, e.g., "English Language,Mathematics,Physics,Chemistry"), 
              `interests` (comma-separated, e.g., "Problem Solving & Logic,Technology & Innovation"), 
              `learning_style`, `state_of_origin`, `gender`, and O'Level grades (e.g., `english_language_grade`, `mathematics_grade`, `physics_grade`).
            - Ensure at least 5 O'Level subjects with valid grades (A1, B2, B3, C4, C5, C6).
        """)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=f"batch_uploader_{st.session_state.uploader_key}")
        if uploaded_file:
            with st.spinner("Validating CSV file..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    logger.info("CSV file uploaded: %s rows, columns: %s", len(df), list(df.columns))
                    if df.empty:
                        st.error("The uploaded CSV file is empty.")
                        logger.error("Empty CSV file uploaded")
                    else:
                        required_columns = [
                            'student_id', 'name', 'utme_score', 'preferred_course', 'utme_subjects',
                            'interests', 'learning_style', 'state_of_origin', 'gender',
                            'english_language_grade', 'mathematics_grade'
                        ]
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        if missing_columns:
                            st.error(f"Missing required columns: {', '.join(missing_columns)}")
                            logger.error("Missing required columns: %s", missing_columns)
                        else:
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            def update_progress(progress):
                                progress_bar.progress(min(progress, 1.0))
                                progress_text.text(f"Processing: {int(progress * 100)}%")

                            course_capacities = get_dynamic_course_capacities(df)
                            logger.info("Starting async CSV processing")
                            admission_results, eligible_courses_df, invalid_rows = asyncio.run(
                                process_with_timeout(
                                    process_csv_applications(df, course_capacities, update_progress)
                                )
                            )
                            logger.info("Async CSV processing completed")

                            if invalid_rows:
                                st.warning(f"Skipped {len(invalid_rows)} invalid rows. Check details below.")
                                st.dataframe(
                                    pd.DataFrame(invalid_rows),
                                    column_config={
                                        'row': 'Row Number',
                                        'student_id': 'Student ID',
                                        'error': 'Error Message'
                                    },
                                    hide_index=True
                                )

                            if not admission_results:
                                st.error("No admission results generated. Check the CSV file for errors and refer to the logs for details.")
                            else:
                                detailed_results, summary_stats, course_breakdown = create_comprehensive_admission_report(admission_results, df, course_capacities)
                                
                                st.subheader("Admission Summary")
                                for key, value in summary_stats.items():
                                    st.write(f"**{key}**: {value}")
                                
                                st.subheader("Course-wise Admission Breakdown")
                                st.dataframe(
                                    course_breakdown,
                                    column_config={
                                        "admitted_course": "Course",
                                        "Students_Admitted": st.column_config.NumberColumn("Students Admitted"),
                                        "Avg_Score": st.column_config.NumberColumn("Average Score", format="%.2f"),
                                        "Min_Score": st.column_config.NumberColumn("Minimum Score", format="%.2f"),
                                        "Max_Score": st.column_config.NumberColumn("Maximum Score", format="%.2f"),
                                        "Capacity": st.column_config.NumberColumn("Capacity"),
                                        "Utilization_Rate": st.column_config.NumberColumn("Utilization Rate (%)", format="%.1f")
                                    },
                                    hide_index=True
                                )
                                
                                st.subheader("Detailed Admission Results")
                                st.dataframe(
                                    detailed_results,
                                    column_config={
                                        "student_id": "Student ID",
                                        "name": "Name",
                                        "utme_score": st.column_config.NumberColumn("UTME Score"),
                                        "preferred_course": "Preferred Course",
                                        "admitted_course": "Admitted Course",
                                        "status": "Admission Status",
                                        "score": st.column_config.NumberColumn("Score", format="%.2f"),
                                        "rank": st.column_config.NumberColumn("Rank"),
                                        "admission_type": "Admission Type",
                                        "reason": "Reason",
                                        "original_preference": "Original Preference",
                                        "recommendation_reason": "Recommendation Reason",
                                        "available_alternatives": st.column_config.NumberColumn("Available Alternatives"),
                                        "suggested_alternatives": "Suggested Alternatives"
                                    },
                                    hide_index=True
                                )
                                
                                recommended_courses_df = detailed_results[['student_id', 'name', 'admitted_course', 'status', 'suggested_alternatives']].copy()
                                recommended_courses_df['recommended_course'] = recommended_courses_df.apply(
                                    lambda row: row['admitted_course'] if row['status'] in ['ADMITTED', 'ALTERNATIVE_ADMISSION']
                                    else (row['suggested_alternatives'][0] if row['suggested_alternatives'] else 'NONE'),
                                    axis=1
                                )
                                
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    eligible_courses_df.to_excel(writer, sheet_name='Eligible Courses', index=False)
                                    recommended_courses_df.to_excel(writer, sheet_name='Recommended Courses', index=False)
                                excel_buffer.seek(0)
                                
                                st.subheader("Download Course Eligibility Report")
                                st.download_button(
                                    label="Download Course Eligibility Excel Report",
                                    data=excel_buffer,
                                    file_name=f"course_eligibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                
                                st.subheader("Download Admission Letters")
                                for result in admission_results:
                                    if result['status'] in ['ADMITTED', 'ALTERNATIVE_ADMISSION']:
                                        student_data = {
                                            'name': result.get('name', 'Student'),
                                            'student_id': result['student_id'],
                                            'admitted_course': result['admitted_course'],
                                            'admission_type': result['admission_type'],
                                            'status': result['status'],
                                            'original_preference': result.get('original_preference', ''),
                                            'recommendation_reason': result.get('recommendation_reason', '')
                                        }
                                        pdf_buffer = generate_admission_letter(student_data)
                                        st.download_button(
                                            label=f"Download Admission Letter for {student_data['name']} ({student_data['student_id']})",
                                            data=pdf_buffer,
                                            file_name=f"admission_letter_{student_data['student_id']}.pdf",
                                            mime="application/pdf"
                                        )
                                
                                csv_buffer = io.StringIO()
                                detailed_results.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)
                                st.download_button(
                                    label="Download Full Admission Report",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"admission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                except Exception as e:
                    logger.error(f"Error processing CSV file: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"Error processing CSV file: {str(e)}. Please check the file format and refer to the logs for details.")
                finally:
                    st.session_state.uploader_key += 1
            st.session_state.uploader_key += 1
        else:
            st.info("Please upload a CSV file to process admissions.")

    with tab3:
        st.header("Analytics & Insights")
        st.write("Upload a CSV file to analyze student demand and capacity utilization.")
        analytics_file = st.file_uploader("Choose a CSV file for analytics", type="csv", key=f"analytics_uploader_{st.session_state.uploader_key}")
        if analytics_file:
            with st.spinner("Processing analytics..."):
                try:
                    df = pd.read_csv(analytics_file)
                    logger.info("Analytics CSV file uploaded: %s rows, columns: %s", len(df), list(df.columns))
                    if df.empty:
                        st.error("The uploaded CSV file is empty.")
                        logger.error("Empty CSV file uploaded for analytics")
                    else:
                        course_capacities = get_dynamic_course_capacities(df)
                        demand_analysis = analyze_student_demand(df)
                        
                        st.subheader("Student Demand Analysis")
                        demand_df = pd.DataFrame.from_dict(demand_analysis, orient='index').reset_index()
                        demand_df.columns = ['Course', 'Primary Demand', 'Secondary Demand', 'Total Estimated Demand', 'Demand Category']
                        st.dataframe(
                            demand_df,
                            column_config={
                                "Course": "Course",
                                "Primary_Demand": st.column_config.NumberColumn("Primary Demand"),
                                "Secondary_Demand": st.column_config.NumberColumn("Secondary Demand"),
                                "Total_Estimated_Demand": st.column_config.NumberColumn("Total Estimated Demand"),
                                "Demand_Category": "Demand Category"
                            },
                            hide_index=True
                        )
                        
                        st.subheader("Capacity Optimization Suggestions")
                        capacity_suggestions = optimize_course_capacities(
                            {k: v['total_estimated_demand'] for k, v in demand_analysis.items()},
                            course_capacities
                        )
                        if capacity_suggestions:
                            suggestions_df = pd.DataFrame.from_dict(capacity_suggestions, orient='index').reset_index()
                            suggestions_df.columns = ['Course', 'Current Capacity', 'Suggested Capacity', 'Reason', 'Priority']
                            st.dataframe(
                                suggestions_df,
                                column_config={
                                    "Course": "Course",
                                    "Current_Capacity": st.column_config.NumberColumn("Current Capacity"),
                                    "Suggested_Capacity": st.column_config.NumberColumn("Suggested Capacity"),
                                    "Reason": "Reason",
                                    "Priority": "Priority"
                                },
                                hide_index=True
                            )
                        
                        if 'admitted_course' in df.columns:
                            st.subheader("Capacity Utilization")
                            admission_results = run_intelligent_admission_algorithm_v2(df.to_dict('records'), course_capacities)
                            utilization_stats = calculate_capacity_utilization(admission_results, course_capacities)
                            utilization_df = pd.DataFrame.from_dict(utilization_stats, orient='index').reset_index()
                            utilization_df.columns = ['Course', 'Capacity', 'Admitted', 'Available', 'Utilization Rate', 'Status']
                            st.dataframe(
                                utilization_df,
                                column_config={
                                    "Course": "Course",
                                    "Capacity": st.column_config.NumberColumn("Capacity"),
                                    "Admitted": st.column_config.NumberColumn("Admitted"),
                                    "Available": st.column_config.NumberColumn("Available"),
                                    "Utilization_Rate": st.column_config.NumberColumn("Utilization Rate (%)", format="%.1f"),
                                    "Status": "Status"
                                },
                                hide_index=True
                            )
                            
                            st.subheader("Utilization Visualization")
                            fig = px.bar(
                                utilization_df,
                                x='Course',
                                y='Utilization_Rate',
                                color='Status',
                                title='Course Capacity Utilization',
                                labels={'Utilization_Rate': 'Utilization Rate (%)'},
                                height=600
                            )
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error in analytics processing: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"Error processing analytics: {str(e)}. Please check the file format and refer to the logs for details.")

if __name__ == "__main__":
    main()
