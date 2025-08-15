import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="University Admission Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Set default capacity for courses not specified
DEFAULT_CAPACITY = 50

def get_dynamic_course_capacities():
    """Get all course capacities including defaults for unspecified courses"""
    dynamic_capacities = COURSE_CAPACITIES.copy()
    
    # Add default capacities for courses not explicitly defined
    for course in course_names:
        if course not in dynamic_capacities:
            dynamic_capacities[course] = DEFAULT_CAPACITY
    
    return dynamic_capacities

def calculate_capacity_utilization(admission_results, course_capacities):
    """Calculate capacity utilization statistics for each course"""
    utilization_stats = {}
    
    # Count admissions by course
    course_admissions = {}
    for result in admission_results:
        if result['status'] in ['ADMITTED', 'ALTERNATIVE_ADMISSION']:
            course = result['admitted_course']
            course_admissions[course] = course_admissions.get(course, 0) + 1
    
    # Calculate utilization for each course
    for course, capacity in course_capacities.items():
        admitted = course_admissions.get(course, 0)
        utilization_rate = (admitted / capacity) * 100 if capacity > 0 else 0
        
        utilization_stats[course] = {
            'capacity': capacity,
            'admitted': admitted,
            'available': capacity - admitted,
            'utilization_rate': utilization_rate,
            'status': 'Full' if admitted >= capacity else 'Available'
        }
    
    return utilization_stats

def optimize_course_capacities(student_demand, current_capacities, total_capacity_limit=None):
    """Suggest optimal course capacity allocation based on student demand"""
    suggestions = {}
    
    # Calculate demand vs capacity ratio for each course
    demand_ratios = {}
    for course, demand in student_demand.items():
        current_capacity = current_capacities.get(course, DEFAULT_CAPACITY)
        demand_ratios[course] = {
            'demand': demand,
            'current_capacity': current_capacity,
            'demand_ratio': demand / current_capacity if current_capacity > 0 else float('inf'),
            'excess_demand': max(0, demand - current_capacity),
            'underutilized': max(0, current_capacity - demand)
        }
    
    # Generate suggestions
    for course, stats in demand_ratios.items():
        if stats['demand_ratio'] > 1.5:  # High demand
            suggested_capacity = min(int(stats['demand'] * 1.2), 500)  # 20% buffer
            suggestions[course] = {
                'current': stats['current_capacity'],
                'suggested': suggested_capacity,
                'reason': f"High demand ({stats['demand']} applicants vs {stats['current_capacity']} capacity)",
                'priority': 'High'
            }
        elif stats['demand_ratio'] < 0.5:  # Low demand
            suggested_capacity = max(int(stats['demand'] * 1.5), 20)  # Minimum 20 slots
            suggestions[course] = {
                'current': stats['current_capacity'],
                'suggested': suggested_capacity,
                'reason': f"Low demand ({stats['demand']} applicants vs {stats['current_capacity']} capacity)",
                'priority': 'Medium'
            }
    
    return suggestions

def analyze_student_demand(df):
    """Analyze student demand patterns from application data"""
    demand_analysis = {}
    
    # Count preferred course selections
    if 'preferred_course' in df.columns:
        preferred_counts = df['preferred_course'].value_counts().to_dict()
        
        # Also analyze secondary interests that could indicate alternative demand
        if 'interests' in df.columns:
            interest_course_mapping = {}
            for _, row in df.iterrows():
                interests = str(row.get('interests', '')).split(',')
                for interest in interests:
                    interest = interest.strip()
                    if interest in interest_categories:
                        for course in interest_categories[interest]:
                            if course not in interest_course_mapping:
                                interest_course_mapping[course] = 0
                            interest_course_mapping[course] += 1
        
        # Combine preferred and interest-based demand
        for course in course_names:
            primary_demand = preferred_counts.get(course, 0)
            secondary_demand = interest_course_mapping.get(course, 0) * 0.3  # Weight secondary interest
            total_demand = primary_demand + secondary_demand
            
            demand_analysis[course] = {
                'primary_demand': primary_demand,
                'secondary_demand': int(secondary_demand),
                'total_estimated_demand': int(total_demand),
                'demand_category': 'High' if total_demand > 100 else 'Medium' if total_demand > 50 else 'Low'
            }
    
    return demand_analysis

@st.cache_data
def load_course_data():
    """Load and return all course-related data"""

    # Common O'Level subjects
    common_subjects = [
        "English Language",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Agricultural Science",
        "Geography",
        "Economics",
        "Further Mathematics",
        "Statistics",
        "Fine Art",
        "Technical Drawing",
        "Introduction to Building Construction",
        "Bricklaying/Blocklaying",
        "Concreting",
        "Joinery",
        "Carpentry",
        "Decorative Painting",
        "Ceramics",
        "Graphics Design",
        "Graphic Printing",
        "Basic Electricity",
        "Introduction to Agricultural Science",
        "Materials and Workshop Process and Machining",
        "Tractor Layout Power Unit Under Carriage and Auto Electricity",
        "Basic Catering and Food Services",
        "Bakery and Confectionaries",
        "Hotel & Catering Craft (Cookery)",
        "Government",
        "Commerce",
        "Accounting",
        "Literature in English",
        "History",
        "CRK",
        "IRK",
        "Social Studies",
    ]

    # Grade mapping
    grade_map = {
        "A1": 1,
        "B2": 2,
        "B3": 3,
        "C4": 4,
        "C5": 5,
        "C6": 6,
        "D7": 7,
        "E8": 8,
        "F9": 9,
    }

    # Course names
    course_names = [
        "Agric Extension & Communication Technology",
        "Agricultural Engineering",
        "Agriculture Resource Economics",
        "Animal Production & Health Services",
        "Applied Geology",
        "Applied Geophysics",
        "Architecture",
        "Biochemistry",
        "Biology",
        "Biomedical Technology",
        "Biotechnology",
        "Building",
        "Civil Engineering",
        "Computer Engineering",
        "Computer Science",
        "Crop Soil & Pest Management",
        "Cyber Security",
        "Ecotourism & Wildlife Management",
        "Electrical / Electronics Engineering",
        "Entrepreneurship",
        "Estate Management",
        "Fisheries & Aquaculture",
        "Food Science & Technology",
        "Forestry & Wood Technology",
        "Human Anatomy",
        "Industrial & Production Engineering",
        "Industrial Chemistry",
        "Industrial Design",
        "Industrial Mathematics",
        "Information & Communication Technology",
        "Information Systems",
        "Information Technology",
        "Marine Science & Technology",
        "Mathematics",
        "Mechanical Engineering",
        "Medical Laboratory Science",
        "Metallurgical & Materials Engineering",
        "Meteorology",
        "Microbiology",
        "Mining Engineering",
        "Physics",
        "Physiology",
        "Quantity Surveying",
        "Remote Sensing & Geoscience Information System",
        "Software Engineering",
        "Statistics",
        "Surveying & Geoinformatics",
        "Textile Design Technology",
        "Urban & Regional Planning",
    ]

    course_groups = {
        "agriculture": [
            "Agric Extension & Communication Technology",
            "Agricultural Engineering",
            "Agriculture Resource Economics",
            "Animal Production & Health Services",
            "Crop Soil & Pest Management",
            "Ecotourism & Wildlife Management",
            "Fisheries & Aquaculture",
            "Food Science & Technology",
            "Forestry & Wood Technology",
        ],
        "engineering": [
            "Agricultural Engineering",
            "Civil Engineering",
            "Computer Engineering",
            "Electrical / Electronics Engineering",
            "Industrial & Production Engineering",
            "Mechanical Engineering",
            "Metallurgical & Materials Engineering",
            "Mining Engineering",
        ],
        "science": [
            "Applied Geology",
            "Applied Geophysics",
            "Biochemistry",
            "Biology",
            "Biomedical Technology",
            "Biotechnology",
            "Industrial Chemistry",
            "Industrial Mathematics",
            "Marine Science & Technology",
            "Mathematics",
            "Meteorology",
            "Microbiology",
            "Physics",
            "Statistics",
        ],
        "technology": [
            "Architecture",
            "Building",
            "Computer Science",
            "Cyber Security",
            "Information & Communication Technology",
            "Information Systems",
            "Information Technology",
            "Software Engineering",
        ],
        "health": ["Human Anatomy", "Medical Laboratory Science", "Physiology"],
        "management": [
            "Entrepreneurship",
            "Estate Management",
            "Quantity Surveying",
            "Surveying & Geoinformatics",
            "Urban & Regional Planning",
        ],
        "design": ["Industrial Design", "Textile Design Technology"],
        "geoscience": ["Remote Sensing & Geoscience Information System"],
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
        "Medicine": {
            "description": "Study of human health, disease diagnosis and treatment",
            "duration": "6 years",
            "career_paths": ["Medical Doctor", "Surgeon", "Specialist", "Medical Researcher"],
            "average_salary": "₦3,000,000 - ₦15,000,000",
            "job_outlook": "Excellent",
            "skills_developed": ["Critical Thinking", "Communication", "Empathy", "Scientific Analysis"]
        }
        # Add more course details as needed
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
        "Hands-on Learner": ["Engineering courses", "Laboratory Sciences", "Building", "Agriculture"],
        "Analytical Thinker": ["Mathematics", "Statistics", "Computer Science", "Applied Sciences"],
        "People-oriented": ["Medicine", "Entrepreneurship", "Extension Services", "Management"]
    }

    # UTME cutoff marks
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
    # Set default cutoff of 180 for other courses
    for course in course_names:
        if course not in cutoff_marks:
            cutoff_marks[course] = 180

    return common_subjects, grade_map, course_names, course_groups, cutoff_marks, course_details, interest_categories, learning_styles

def process_csv_applications(df, course_capacities):
    """Process batch applications using advanced ML algorithms"""
    results = []
    
    # Prepare data for ML processing
    processed_students = []
    
    for idx, row in df.iterrows():
        student_data = {
            'student_id': row.get('student_id', f'STU_{idx:04d}'),
            'name': row.get('name', f'Student_{idx}'),
            'utme_score': row.get('utme_score', 0),
            'preferred_course': row.get('preferred_course', ''),
            'interests': row.get('interests', '').split(',') if row.get('interests') else [],
            'learning_style': row.get('learning_style', 'Analytical Thinker'),
            'olevel_subjects': {},
            'state_of_origin': row.get('state_of_origin', ''),
            'gender': row.get('gender', ''),
            'age': row.get('age', 18)
        }
        
        # Parse O'Level subjects from CSV
        for subject in common_subjects:
            grade_col = f"{subject.lower().replace(' ', '_').replace('/', '_')}_grade"
            if grade_col in row and pd.notna(row[grade_col]):
                student_data['olevel_subjects'][subject] = grade_map.get(row[grade_col], 9)
        
        processed_students.append(student_data)
    
    # Run advanced admission algorithm
    admission_results = run_intelligent_admission_algorithm_v2(processed_students, course_capacities)
    
    return admission_results

def calculate_comprehensive_score(student, course, base_score):
    """Calculate comprehensive score including diversity factors and performance predictors"""
    
    # Base academic score (60% weight)
    final_score = base_score * 0.6
    
    # Interest alignment bonus (20% weight)
    interest_bonus = 0
    if student['interests']:
        for interest in student['interests']:
            if interest.strip() in interest_categories:
                if course in interest_categories[interest.strip()]:
                    interest_bonus += 0.05
    final_score += min(interest_bonus, 0.2)
    
    # Learning style compatibility (10% weight)
    learning_bonus = 0
    if student['learning_style'] in learning_styles:
        compatible_courses = learning_styles[student['learning_style']]
        if any(course_type in course for course_type in ['Engineering', 'Science', 'Technology']):
            if student['learning_style'] in ['Analytical Thinker', 'Visual Learner']:
                learning_bonus = 0.1
    final_score += learning_bonus
    
    # Performance prediction based on O'Level pattern (10% weight)
    olevel_performance = calculate_olevel_performance_indicator(student['olevel_subjects'])
    final_score += olevel_performance * 0.1
    
    return min(final_score, 1.0)

def calculate_olevel_performance_indicator(olevel_subjects):
    """Calculate performance indicator based on O'Level results pattern"""
    if not olevel_subjects:
        return 0
    
    grades = list(olevel_subjects.values())
    if not grades:
        return 0
    
    # Calculate weighted performance
    excellent_count = sum(1 for grade in grades if grade <= 2)  # A1, B2
    good_count = sum(1 for grade in grades if grade <= 4)  # Up to C4
    total_subjects = len(grades)
    
    # Performance indicator (0-1 scale)
    performance = (excellent_count * 1.0 + (good_count - excellent_count) * 0.7) / total_subjects
    return min(performance, 1.0)

def calculate_course_similarity(course1, course2, course_groups):
    """Calculate similarity score between two courses based on their groups and requirements"""
    similarity_score = 0.0
    
    # Check if courses are in the same group
    for group_name, courses in course_groups.items():
        if course1 in courses and course2 in courses:
            similarity_score += 0.4  # High similarity for same group
            break
    
    # Check for related groups (e.g., engineering and technology)
    related_groups = {
        'engineering': ['technology', 'science'],
        'technology': ['engineering', 'science'],
        'science': ['engineering', 'technology', 'health'],
        'health': ['science'],
        'management': ['design'],
        'design': ['management']
    }
    
    course1_groups = [group for group, courses in course_groups.items() if course1 in courses]
    course2_groups = [group for group, courses in course_groups.items() if course2 in courses]
    
    for group1 in course1_groups:
        for group2 in course2_groups:
            if group1 != group2 and group2 in related_groups.get(group1, []):
                similarity_score += 0.2  # Medium similarity for related groups
    
    # Check for keyword similarity in course names
    course1_words = set(course1.lower().split())
    course2_words = set(course2.lower().split())
    common_words = course1_words.intersection(course2_words)
    
    if common_words:
        similarity_score += len(common_words) * 0.1  # Small bonus for common words
    
    return min(similarity_score, 1.0)

def find_smart_alternatives(student_data, student_course_matrix, course_capacities, course_admission_counts, max_alternatives=5):
    """Find intelligent alternative courses using multiple criteria"""
    student_id = student_data['student_id']
    preferred_course = student_data['preferred_course']
    interests = student_data.get('interests', [])
    
    # Find student's course scores
    student_row = next((row for row in student_course_matrix if row['student_id'] == student_id), None)
    if not student_row:
        return []
    
    alternatives = []
    
    for course in course_names:
        if (course != preferred_course and 
            course in course_capacities and
            course_admission_counts.get(course, 0) < course_capacities[course] and
            student_row[course]['eligible']):
            
            # Calculate alternative score based on multiple factors
            base_score = student_row[course]['score']
            
            # Factor 1: Course similarity to preferred course
            similarity_score = calculate_course_similarity(preferred_course, course, course_groups)
            
            # Factor 2: Interest alignment
            interest_alignment = 0.0
            for interest in interests:
                if interest.strip() in interest_categories:
                    if course in interest_categories[interest.strip()]:
                        interest_alignment += 0.2
            interest_alignment = min(interest_alignment, 0.6)
            
            # Factor 3: Capacity availability (prefer courses with more available slots)
            available_slots = course_capacities[course] - course_admission_counts.get(course, 0)
            capacity_factor = min(available_slots / course_capacities[course], 1.0) * 0.1
            
            # Factor 4: Course demand (prefer less competitive courses for better admission chances)
            course_demand = course_admission_counts.get(course, 0)
            demand_factor = (1 - (course_demand / course_capacities[course])) * 0.1
            
            # Calculate comprehensive alternative score
            alternative_score = (
                base_score * 0.4 +  # Academic fit (40%)
                similarity_score * 0.3 +  # Course similarity (30%)
                interest_alignment * 0.2 +  # Interest alignment (20%)
                capacity_factor * 0.05 +  # Capacity availability (5%)
                demand_factor * 0.05  # Demand factor (5%)
            )
            
            alternatives.append({
                'course': course,
                'score': alternative_score,
                'base_score': base_score,
                'similarity_score': similarity_score,
                'interest_alignment': interest_alignment,
                'available_slots': available_slots,
                'recommendation_reason': generate_alternative_reason(
                    preferred_course, course, similarity_score, interest_alignment
                )
            })
    
    # Sort by comprehensive score and return top alternatives
    alternatives.sort(key=lambda x: x['score'], reverse=True)
    return alternatives[:max_alternatives]

def generate_alternative_reason(preferred_course, alternative_course, similarity_score, interest_alignment):
    """Generate human-readable reason for alternative course recommendation"""
    reasons = []
    
    if similarity_score > 0.3:
        reasons.append(f"closely related to {preferred_course}")
    
    if interest_alignment > 0.3:
        reasons.append("matches your interests")
    
    # Check for specific course relationships
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
    """Advanced ML-based admission algorithm with optimization"""
    results = []
    course_admission_counts = {course: 0 for course in course_capacities.keys()}
    
    # Phase 1: Calculate comprehensive scores for all student-course combinations
    student_course_matrix = []
    
    for student in students:
        student_row = {'student_id': student['student_id'], 'student_data': student}
        
        for course in course_names:
            # Get base eligibility and score
            prediction = predict_placement_enhanced(
                student['utme_score'],
                student['olevel_subjects'],
                student['interests'],
                student['learning_style']
            )
            
            base_score = 0
            eligible = False
            
            if 'all_eligible' in prediction and not prediction['all_eligible'].empty:
                course_match = prediction['all_eligible'][prediction['all_eligible']['course'] == course]
                if not course_match.empty:
                    base_score = course_match.iloc[0]['score']
                    eligible = True
            
            # Calculate comprehensive score
            if eligible:
                comprehensive_score = calculate_comprehensive_score(student, course, base_score)
                
                # Preferred course bonus
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
    
    # Phase 2: Optimized admission allocation using priority-based assignment
    admitted_students = set()
    
    # First pass: Admit top candidates to their preferred courses
    for course in course_names:
        if course not in course_capacities:
            continue
            
        capacity = course_capacities[course]
        candidates = []
        
        for student_row in student_course_matrix:
            student_id = student_row['student_id']
            student_data = student_row['student_data']
            
            if (student_id not in admitted_students and 
                student_row[course]['eligible'] and 
                student_data['preferred_course'] == course):
                
                candidates.append({
                    'student_id': student_id,
                    'student_data': student_data,
                    'score': student_row[course]['score'],
                    'course': course
                })
        
        # Sort by score and admit top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for i, candidate in enumerate(candidates[:capacity]):
            results.append({
                'student_id': candidate['student_id'],
                'admitted_course': course,
                'status': 'ADMITTED',
                'score': candidate['score'],
                'rank': i + 1,
                'admission_type': 'PREFERRED'
            })
            admitted_students.add(candidate['student_id'])
            course_admission_counts[course] += 1
    
    # Second pass: Fill remaining slots with best available candidates
    for course in course_names:
        if course not in course_capacities:
            continue
            
        remaining_capacity = course_capacities[course] - course_admission_counts[course]
        
        if remaining_capacity > 0:
            candidates = []
            
            for student_row in student_course_matrix:
                student_id = student_row['student_id']
                
                if (student_id not in admitted_students and 
                    student_row[course]['eligible']):
                    
                    candidates.append({
                        'student_id': student_id,
                        'student_data': student_row['student_data'],
                        'score': student_row[course]['score'],
                        'course': course
                    })
            
            # Sort by score and admit remaining candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            for i, candidate in enumerate(candidates[:remaining_capacity]):
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
    
    remaining_students = []
    for student_row in student_course_matrix:
        student_id = student_row['student_id']
        student_data = student_row['student_data']
        
        if student_id not in admitted_students:
            remaining_students.append((student_row, student_data))
    
    # Sort remaining students by their best alternative score to prioritize stronger candidates
    remaining_students.sort(key=lambda x: max([
        x[0][course]['score'] for course in course_names 
        if x[0][course]['eligible'] and course != x[1]['preferred_course']
    ] + [0]), reverse=True)
    
    for student_row, student_data in remaining_students:
        student_id = student_data['student_id']
        
        if student_id not in admitted_students:
            # Find smart alternative courses
            smart_alternatives = find_smart_alternatives(
                student_data, student_course_matrix, course_capacities, course_admission_counts
            )
            
            if smart_alternatives:
                # Try to admit to the best available alternative
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
                        break
                else:
                    # No capacity available in any alternative
                    results.append({
                        'student_id': student_id,
                        'admitted_course': 'NONE',
                        'status': 'WAITLISTED',
                        'score': 0,
                        'reason': 'Qualified but no capacity available in suitable alternatives',
                        'admission_type': 'WAITLISTED',
                        'suggested_alternatives': [alt['course'] for alt in smart_alternatives[:3]]
                    })
            else:
                # Check if student meets minimum university requirements
                min_requirements_met = check_minimum_university_requirements(student_data)
                
                results.append({
                    'student_id': student_id,
                    'admitted_course': 'NONE',
                    'status': 'NOT_QUALIFIED',
                    'score': 0,
                    'reason': 'No available course matches qualifications' if min_requirements_met else 'Does not meet minimum university requirements',
                    'admission_type': 'REJECTED'
                })
    
    return results

def check_minimum_university_requirements(student_data):
    """Check if student meets minimum university admission requirements"""
    utme_score = student_data.get('utme_score', 0)
    olevel_subjects = student_data.get('olevel_subjects', {})
    
    # Minimum UTME score requirement
    if utme_score < 160:
        return False
    
    # Minimum O'Level requirements
    required_subjects = ['English Language', 'Mathematics']
    credits_count = 0
    
    for subject, grade in olevel_subjects.items():
        if grade <= 6:  # C6 and above
            credits_count += 1
            
    # Must have English and Mathematics at credit level
    for req_subject in required_subjects:
        if req_subject not in olevel_subjects or olevel_subjects[req_subject] > 6:
            return False
    
    # Must have at least 5 credits
    if credits_count < 5:
        return False
    
    return True

def find_alternative_courses_v2(student, student_course_matrix, course_capacities, admitted_courses):
    """Enhanced alternative course finder with capacity consideration"""
    student_row = next((row for row in student_course_matrix if row['student_id'] == student['student_id']), None)
    
    if not student_row:
        return []
    
    alternatives = []
    
    for course in course_names:
        if (course != student['preferred_course'] and 
            course in course_capacities and
            course not in admitted_courses and
            student_row[course]['eligible']):
            
            alternatives.append({
                'course': course,
                'score': student_row[course]['score'],
                'capacity_available': course_capacities[course] > admitted_courses.get(course, 0)
            })
    
    # Sort by score and filter only courses with available capacity
    alternatives = [alt for alt in alternatives if alt['capacity_available']]
    alternatives.sort(key=lambda x: x['score'], reverse=True)
    
    return alternatives[:5]  # Return top 5 alternatives

def run_intelligent_admission_algorithm(students, course_capacities):
    """Advanced ML-based admission algorithm"""
    results = []
    course_admissions = {course: 0 for course in course_capacities.keys()}
    
    # Calculate scores for all students and courses
    student_course_scores = []
    
    for student in students:
        student_scores = []
        
        for course in course_names:
            # Get eligibility and score
            prediction = predict_placement_enhanced(
                student['utme_score'],
                student['olevel_subjects'],
                student['interests'],
                student['learning_style']
            )
            
            if course in prediction.get('all_eligible', pd.DataFrame()).get('course', []).tolist():
                course_score_row = prediction['all_eligible'][prediction['all_eligible']['course'] == course]
                if not course_score_row.empty:
                    score = course_score_row.iloc[0]['score']
                else:
                    score = 0
            else:
                score = 0
            
            # Boost score if it's preferred course
            if course == student['preferred_course']:
                score *= 1.2
            
            student_scores.append({
                'student_id': student['student_id'],
                'course': course,
                'score': score,
                'eligible': score > 0
            })
        
        student_course_scores.extend(student_scores)
    
    # Convert to DataFrame for easier processing
    scores_df = pd.DataFrame(student_course_scores)
    
    # Admission allocation using Hungarian algorithm approach
    for course in course_names:
        capacity = course_capacities.get(course, DEFAULT_CAPACITY)
        course_candidates = scores_df[
            (scores_df['course'] == course) & 
            (scores_df['eligible'] == True) & 
            (scores_df['score'] > 0)
        ].sort_values('score', ascending=False)
        
        admitted_count = 0
        for _, candidate in course_candidates.iterrows():
            if admitted_count < capacity:
                # Check if student hasn't been admitted elsewhere
                student_id = candidate['student_id']
                already_admitted = any(
                    r['student_id'] == student_id and r['status'] == 'ADMITTED' 
                    for r in results
                )
                
                if not already_admitted:
                    results.append({
                        'student_id': student_id,
                        'admitted_course': course,
                        'status': 'ADMITTED',
                        'score': candidate['score'],
                        'rank': admitted_count + 1
                    })
                    admitted_count += 1
    
    # Handle students not admitted to preferred courses
    admitted_students = {r['student_id'] for r in results if r['status'] == 'ADMITTED'}
    
    for student in students:
        if student['student_id'] not in admitted_students:
            # Find alternative courses
            alternatives = find_alternative_courses(student, scores_df)
            
            if alternatives:
                best_alternative = alternatives[0]
                results.append({
                    'student_id': student['student_id'],
                    'admitted_course': best_alternative['course'],
                    'status': 'ALTERNATIVE_ADMISSION',
                    'score': best_alternative['score'],
                    'original_preference': student['preferred_course']
                })
            else:
                results.append({
                    'student_id': student['student_id'],
                    'admitted_course': 'NONE',
                    'status': 'NOT_QUALIFIED',
                    'score': 0,
                    'reason': 'Does not meet minimum requirements'
                })
    
    return results

def find_alternative_courses(student, scores_df):
    """Find alternative courses for students not admitted to preferred course"""
    student_scores = scores_df[
        (scores_df['student_id'] == student['student_id']) & 
        (scores_df['eligible'] == True) & 
        (scores_df['score'] > 0)
    ].sort_values('score', ascending=False)
    
    alternatives = []
    for _, row in student_scores.iterrows():
        if row['course'] != student['preferred_course']:
            alternatives.append({
                'course': row['course'],
                'score': row['score']
            })
    
    return alternatives[:3]  # Return top 3 alternatives

def create_sample_csv():
    """Create a sample CSV file for demonstration"""
    sample_data = {
        'student_id': [f'STU_{i:04d}' for i in range(1, 21)],
        'name': [f'Student {i}' for i in range(1, 21)],
        'utme_score': np.random.randint(150, 350, 20),
        'preferred_course': np.random.choice(['Computer Science', 'Civil Engineering', 'Electrical / Electronics Engineering'], 20),
        'interests': ['Problem Solving & Logic,Technology & Innovation'] * 10 + ['Building & Construction,Research & Analysis'] * 10,
        'learning_style': np.random.choice(['Visual Learner', 'Analytical Thinker', 'Hands-on Learner'], 20),
        'english_language_grade': np.random.choice(['A1', 'B2', 'B3', 'C4', 'C5'], 20),
        'mathematics_grade': np.random.choice(['A1', 'B2', 'B3', 'C4', 'C5'], 20),
        'physics_grade': np.random.choice(['A1', 'B2', 'B3', 'C4', 'C5', 'C6'], 20),
        'chemistry_grade': np.random.choice(['A1', 'B2', 'B3', 'C4', 'C5', 'C6'], 20),
        'biology_grade': np.random.choice(['A1', 'B2', 'B3', 'C4', 'C5', 'C6'], 20),
    }
    
    return pd.DataFrame(sample_data)

def download_csv(df, filename):
    """Create download link for CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

@st.cache_data
def safe_load_data():
    """Safely load data with error handling"""
    try:
        return load_course_data()
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return None, None, None, None, None, None, None, None

# Title and description
st.title("🎓 University Admission Management System")
st.markdown(
    "**Intelligent batch processing system for university admissions with ML-powered course allocation**"
)

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Individual Recommendation", "Batch Admission Processing"],
    help="Choose between individual student recommendations or batch processing for admissions"
)

# Initialize session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

try:
    data_result = safe_load_data()
    if data_result[0] is not None:
        common_subjects, grade_map, course_names, course_groups, cutoff_marks, course_details, interest_categories, learning_styles = data_result
    else:
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize application: {str(e)}")
    st.stop()

if mode == "Batch Admission Processing":
    st.header("📊 Batch Admission Processing")
    
    st.subheader("🎯 Course Capacity Configuration")
    
    # Capacity management tabs
    capacity_tab1, capacity_tab2, capacity_tab3 = st.tabs(["Manual Configuration", "Smart Suggestions", "Capacity Analytics"])
    
    with capacity_tab1:
        st.markdown("**Configure course capacities manually:**")
        
        # Organize courses by category for better UX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔧 Engineering & Technology:**")
            engineering_courses = [
                "Computer Science", "Civil Engineering", "Electrical / Electronics Engineering", 
                "Mechanical Engineering", "Computer Engineering", "Software Engineering",
                "Industrial & Production Engineering"
            ]
            for course in engineering_courses:
                if course in course_names:
                    COURSE_CAPACITIES[course] = st.number_input(
                        f"{course}",
                        min_value=10,
                        max_value=500,
                        value=COURSE_CAPACITIES.get(course, DEFAULT_CAPACITY),
                        key=f"capacity_{course}",
                        help=f"Set admission capacity for {course}"
                    )
        
        with col2:
            st.markdown("**🧬 Science & Health:**")
            science_courses = [
                "Human Anatomy", "Medical Laboratory Science", "Physiology", 
                "Biochemistry", "Mathematics", "Physics", "Statistics",
                "Biology", "Chemistry", "Microbiology"
            ]
            for course in science_courses:
                if course in course_names:
                    COURSE_CAPACITIES[course] = st.number_input(
                        f"{course}",
                        min_value=10,
                        max_value=500,
                        value=COURSE_CAPACITIES.get(course, DEFAULT_CAPACITY),
                        key=f"capacity_{course}",
                        help=f"Set admission capacity for {course}"
                    )
        
        with col3:
            st.markdown("**🏗️ Other Disciplines:**")
            other_courses = [
                "Architecture", "Building", "Information Systems", 
                "Information Technology", "Cyber Security", "Industrial Design",
                "Estate Management", "Quantity Surveying"
            ]
            for course in other_courses:
                if course in course_names:
                    COURSE_CAPACITIES[course] = st.number_input(
                        f"{course}",
                        min_value=10,
                        max_value=500,
                        value=COURSE_CAPACITIES.get(course, DEFAULT_CAPACITY),
                        key=f"capacity_{course}",
                        help=f"Set admission capacity for {course}"
                    )
        
        # Quick capacity adjustment tools
        st.markdown("**⚡ Quick Adjustments:**")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("Set All Engineering to 100"):
                for course in engineering_courses:
                    if course in COURSE_CAPACITIES:
                        COURSE_CAPACITIES[course] = 100
                st.rerun()
        
        with quick_col2:
            if st.button("Set All Science to 80"):
                for course in science_courses:
                    if course in COURSE_CAPACITIES:
                        COURSE_CAPACITIES[course] = 80
                st.rerun()
        
        with quick_col3:
            total_capacity = sum(COURSE_CAPACITIES.values())
            st.metric("Total University Capacity", total_capacity)
    
    with capacity_tab2:
        st.markdown("**AI-powered capacity suggestions based on demand analysis:**")
        
        # File upload for demand analysis
        demand_file = st.file_uploader(
            "Upload student applications for demand analysis",
            type=['csv'],
            key="demand_analysis_file",
            help="Upload CSV to analyze student demand patterns and get capacity suggestions"
        )
        
        if demand_file is not None:
            try:
                demand_df = pd.read_csv(demand_file)
                
                # Analyze demand
                demand_analysis = analyze_student_demand(demand_df)
                capacity_suggestions = optimize_course_capacities(
                    {course: stats['total_estimated_demand'] for course, stats in demand_analysis.items()},
                    COURSE_CAPACITIES
                )
                
                if capacity_suggestions:
                    st.subheader("📈 Capacity Optimization Suggestions")
                    
                    for course, suggestion in capacity_suggestions.items():
                        with st.expander(f"📋 {course} - {suggestion['priority']} Priority"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Capacity", suggestion['current'])
                            with col2:
                                st.metric("Suggested Capacity", suggestion['suggested'])
                            with col3:
                                change = suggestion['suggested'] - suggestion['current']
                                st.metric("Change", f"{change:+d}")
                            
                            st.write(f"**Reason:** {suggestion['reason']}")
                            
                            if st.button(f"Apply Suggestion for {course}", key=f"apply_{course}"):
                                COURSE_CAPACITIES[course] = suggestion['suggested']
                                st.success(f"Updated {course} capacity to {suggestion['suggested']}")
                                st.rerun()
                
                # Demand visualization
                st.subheader("📊 Student Demand Analysis")
                
                demand_data = []
                for course, stats in demand_analysis.items():
                    demand_data.append({
                        'Course': course,
                        'Primary Demand': stats['primary_demand'],
                        'Secondary Demand': stats['secondary_demand'],
                        'Total Demand': stats['total_estimated_demand'],
                        'Current Capacity': COURSE_CAPACITIES.get(course, DEFAULT_CAPACITY),
                        'Category': stats['demand_category']
                    })
                
                demand_viz_df = pd.DataFrame(demand_data)
                
                # Create demand vs capacity chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Student Demand',
                    x=demand_viz_df['Course'],
                    y=demand_viz_df['Total Demand'],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Current Capacity',
                    x=demand_viz_df['Course'],
                    y=demand_viz_df['Current Capacity'],
                    marker_color='orange'
                ))
                
                fig.update_layout(
                    title='Student Demand vs Current Capacity',
                    xaxis_title='Course',
                    yaxis_title='Number of Students',
                    barmode='group',
                    height=500
                )
                fig.update_xaxis(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing demand: {str(e)}")
        else:
            st.info("Upload a CSV file with student applications to get AI-powered capacity suggestions")
    
    with capacity_tab3:
        st.markdown("**Real-time capacity analytics and utilization tracking:**")
        
        # Current capacity overview
        current_capacities = get_dynamic_course_capacities()
        
        # Create capacity overview visualization
        capacity_data = []
        for course, capacity in current_capacities.items():
            category = 'Engineering' if any(eng in course for eng in ['Engineering', 'Computer', 'Software']) else \
                      'Science' if any(sci in course for sci in ['Science', 'Mathematics', 'Physics', 'Chemistry', 'Biology']) else \
                      'Health' if any(health in course for health in ['Anatomy', 'Medical', 'Physiology']) else 'Other'
            
            capacity_data.append({
                'Course': course,
                'Capacity': capacity,
                'Category': category
            })
        
        capacity_df = pd.DataFrame(capacity_data)
        
        # Capacity by category
        category_summary = capacity_df.groupby('Category')['Capacity'].sum().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                category_summary, 
                values='Capacity', 
                names='Category',
                title='Capacity Distribution by Category'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                capacity_df.head(15), 
                x='Course', 
                y='Capacity',
                color='Category',
                title='Top 15 Courses by Capacity'
            )
            fig_bar.update_xaxis(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Capacity statistics
        st.subheader("📈 Capacity Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            total_capacity = capacity_df['Capacity'].sum()
            st.metric("Total University Capacity", f"{total_capacity:,}")
        
        with stat_col2:
            avg_capacity = capacity_df['Capacity'].mean()
            st.metric("Average Course Capacity", f"{avg_capacity:.0f}")
        
        with stat_col3:
            max_capacity_course = capacity_df.loc[capacity_df['Capacity'].idxmax()]
            st.metric("Largest Course", f"{max_capacity_course['Course'][:20]}...", f"{max_capacity_course['Capacity']}")
        
        with stat_col4:
            min_capacity_course = capacity_df.loc[capacity_df['Capacity'].idxmin()]
            st.metric("Smallest Course", f"{min_capacity_course['Course'][:20]}...", f"{min_capacity_course['Capacity']}")

    # File upload section
    st.subheader("📁 Upload Student Applications")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with student applications",
            type=['csv'],
            help="Upload a CSV file containing student information including UTME scores, O'Level results, and preferences"
        )
    
    with col2:
        st.markdown("**Need a sample file?**")
        sample_df = create_sample_csv()
        st.markdown(download_csv(sample_df, "sample_applications.csv"), unsafe_allow_html=True)
        
        if st.button("Preview Sample Data"):
            st.dataframe(sample_df.head())
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} student applications")
            
            # Display file preview
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(df.head(10))
            
            # Process applications
            if st.button("🚀 Process Admissions", type="primary"):
                with st.spinner("Processing admissions using ML algorithms..."):
                    # Use dynamic capacities
                    dynamic_capacities = get_dynamic_course_capacities()
                    admission_results = process_csv_applications(df, dynamic_capacities)
                
                # Display results
                st.header("📊 Admission Results")
                
                results_df = pd.DataFrame(admission_results)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    admitted_count = len(results_df[results_df['status'] == 'ADMITTED'])
                    st.metric("✅ Direct Admissions", admitted_count)
                
                with col2:
                    alternative_count = len(results_df[results_df['status'] == 'ALTERNATIVE_ADMISSION'])
                    st.metric("🔄 Alternative Admissions", alternative_count)
                
                with col3:
                    rejected_count = len(results_df[results_df['status'] == 'NOT_QUALIFIED'])
                    st.metric("❌ Not Qualified", rejected_count)
                
                with col4:
                    total_admitted = admitted_count + alternative_count
                    admission_rate = (total_admitted / len(df)) * 100
                    st.metric("📈 Admission Rate", f"{admission_rate:.1f}%")
                
                # Capacity utilization analysis
                st.subheader("🎯 Capacity Utilization Analysis")
                
                utilization_stats = calculate_capacity_utilization(admission_results, dynamic_capacities)
                
                # Create utilization visualization
                util_data = []
                for course, stats in utilization_stats.items():
                    if stats['admitted'] > 0:  # Only show courses with admissions
                        util_data.append({
                            'Course': course,
                            'Capacity': stats['capacity'],
                            'Admitted': stats['admitted'],
                            'Available': stats['available'],
                            'Utilization Rate': stats['utilization_rate'],
                            'Status': stats['status']
                        })
                
                if util_data:
                    util_df = pd.DataFrame(util_data)
                    
                    # Utilization rate chart
                    fig_util = px.bar(
                        util_df.sort_values('Utilization Rate', ascending=False),
                        x='Course',
                        y='Utilization Rate',
                        color='Status',
                        title='Course Capacity Utilization Rates',
                        labels={'Utilization Rate': 'Utilization Rate (%)'}
                    )
                    fig_util.update_xaxis(tickangle=45)
                    fig_util.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Full Capacity")
                    st.plotly_chart(fig_util, use_container_width=True)
                    
                    # Utilization table
                    st.dataframe(util_df, use_container_width=True)

                # Course-wise admission breakdown
                st.subheader("📊 Course-wise Admission Breakdown")
                
                course_breakdown = results_df[results_df['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])].groupby('admitted_course').size().reset_index(name='count')
                
                if not course_breakdown.empty:
                    fig = px.bar(
                        course_breakdown,
                        x='admitted_course',
                        y='count',
                        title="Students Admitted by Course",
                        labels={'admitted_course': 'Course', 'count': 'Number of Students'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Admission Results")
                
                # Merge with original student data
                detailed_results = results_df.merge(
                    df[['student_id', 'name', 'utme_score', 'preferred_course']], 
                    on='student_id', 
                    how='left'
                )
                
                # Format the results table
                display_columns = ['name', 'student_id', 'utme_score', 'preferred_course', 'admitted_course', 'status', 'score']
                if 'rank' in detailed_results.columns:
                    display_columns.append('rank')
                
                st.dataframe(
                    detailed_results[display_columns].sort_values('score', ascending=False),
                    use_container_width=True
                )
                
                # Download results
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    admitted_students = detailed_results[detailed_results['status'].isin(['ADMITTED', 'ALTERNATIVE_ADMISSION'])]
                    st.markdown(
                        download_csv(admitted_students, f"admitted_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    rejected_students = detailed_results[detailed_results['status'] == 'NOT_QUALIFIED']
                    st.markdown(
                        download_csv(rejected_students, f"rejected_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                        unsafe_allow_html=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the required columns: student_id, name, utme_score, preferred_course, interests, learning_style, and O'Level grade columns")

else:
    # ... existing code for individual recommendations ...
    
    # Course requirements (simplified version for demo)
    @st.cache_data
    def get_course_requirements():
        """Return simplified course requirements"""
        return {
            "Computer Science": {
                "mandatory": ["English Language", "Mathematics", "Physics"],
                "or_groups": [],
                "optional": {
                    2: [
                        "Biology",
                        "Chemistry",
                        "Agricultural Science",
                        "Economics",
                        "Geography",
                    ]
                },
                "thresholds": {},
                "required_credit_count": 5,
            },
            "Civil Engineering": {
                "mandatory": ["English Language", "Physics", "Chemistry", "Mathematics"],
                "or_groups": [],
                "optional": {1: ["Biology", "Further Mathematics", "Geography"]},
                "thresholds": {},
                "required_credit_count": 5,
            },
            "Medicine": {
                "mandatory": [
                    "English Language",
                    "Mathematics",
                    "Biology",
                    "Chemistry",
                    "Physics",
                ],
                "or_groups": [],
                "optional": {},
                "thresholds": {},
                "required_credit_count": 5,
            },
            # Add more courses as needed
        }

    def get_related_courses_with_weights(selected_interests, interest_categories, learning_style, learning_styles):
        """Get related courses based on multiple interests and learning style with weights"""
        course_weights = defaultdict(float)
        
        # Weight courses based on selected interests
        for interest in selected_interests:
            if interest in interest_categories:
                for course in interest_categories[interest]:
                    course_weights[course] += 0.3  # Interest weight
        
        # Weight courses based on learning style
        if learning_style in learning_styles:
            for course in learning_styles[learning_style]:
                if "courses" in course:  # Handle generic categories
                    continue
                course_weights[course] += 0.2  # Learning style weight
        
        return dict(course_weights)

    def is_eligible(applicant_subjects, course, parsed_req):
        """Check if applicant meets O'Level requirements for a course"""
        if course not in parsed_req:
            # Simplified eligibility check for courses not in detailed requirements
            required_subjects = ["English Language", "Mathematics"]
            return all(
                sub in applicant_subjects and applicant_subjects[sub] <= 6
                for sub in required_subjects
            )

        p = parsed_req[course]
        thresh = p["thresholds"]
        good_credit = {s for s in applicant_subjects if applicant_subjects[s] <= 6}

        if len(good_credit) < p["required_credit_count"]:
            return False

        # Check mandatory subjects
        for s in p["mandatory"]:
            if s not in applicant_subjects or applicant_subjects[s] > thresh.get(s, 6):
                return False

        # Check OR groups
        for group in p["or_groups"]:
            if not any(
                sub in applicant_subjects and applicant_subjects[sub] <= thresh.get(sub, 6)
                for sub in group
            ):
                return False

        # Check optional subjects
        for num, group in p["optional"].items():
            count = sum(
                1
                for sub in group
                if sub in applicant_subjects
                and applicant_subjects[sub] <= thresh.get(sub, 6)
            )
            if count < num:
                return False

        return True

    def compute_grade_sum(applicant_subjects, course, parsed_req):
        """Compute sum of grades and count of subjects meeting course requirements"""
        if course not in parsed_req:
            # Simplified calculation for courses not in detailed requirements
            relevant_subjects = [
                "English Language",
                "Mathematics",
                "Physics",
                "Chemistry",
                "Biology",
            ]
            grade_sum = sum(
                applicant_subjects.get(sub, 9)
                for sub in relevant_subjects
                if sub in applicant_subjects
            )
            count = len([sub for sub in relevant_subjects if sub in applicant_subjects])
            return grade_sum, max(count, 1)

        p = parsed_req[course]
        thresh = p["thresholds"]
        grade_sum = 0
        count = 0

        # Mandatory subjects
        for s in p["mandatory"]:
            if s in applicant_subjects and applicant_subjects[s] <= thresh.get(s, 6):
                grade_sum += applicant_subjects[s]
                count += 1

        # OR groups (take min grade)
        for group in p["or_groups"]:
            candidates = [
                applicant_subjects[sub]
                for sub in group
                if sub in applicant_subjects
                and applicant_subjects[sub] <= thresh.get(sub, 6)
            ]
            if candidates:
                grade_sum += min(candidates)
                count += 1

        # Optional subjects (take lowest num grades)
        for num, group in p["optional"].items():
            candidates = [
                applicant_subjects[sub]
                for sub in group
                if sub in applicant_subjects
                and applicant_subjects[sub] <= thresh.get(sub, 6)
            ]
            candidates.sort()
            if len(candidates) >= num:
                grade_sum += sum(candidates[:num])
                count += num

        return grade_sum, max(count, 1)

    def compute_enhanced_score(utme_score, grade_sum, count, course_weight, base_interest_bonus=0.15):
        """Compute enhanced applicant score for a course with interest weighting"""
        if count == 0:
            return 0

        normalized_utme = utme_score / 400  # Normalize UTME score (0-400)
        average_grade = grade_sum / count
        normalized_grade = (9 - average_grade) / 8  # Normalize grades (lower is better)
        
        # Base academic score
        base_score = 0.5 * normalized_utme + 0.5 * normalized_grade
        
        # Apply interest weighting
        interest_bonus = course_weight * base_interest_bonus
        final_score = base_score + interest_bonus

        return max(0, min(final_score, 1 + base_interest_bonus))

    def predict_placement_enhanced(utme_score, olevel_subjects, selected_interests, learning_style):
        """Enhanced prediction with multiple interests and learning style consideration"""
        parsed_req = get_course_requirements()

        if not isinstance(utme_score, (int, float)) or utme_score < 0 or utme_score > 400:
            return {
                "predicted_program": "UNASSIGNED",
                "score": 0,
                "reason": "Invalid UTME score",
            }

        # Get course weights based on interests and learning style
        course_weights = get_related_courses_with_weights(selected_interests, interest_categories, learning_style, learning_styles)

        results = []

        for course in course_names:
            eligible = (
                is_eligible(olevel_subjects, course, parsed_req)
                and utme_score >= cutoff_marks[course]
            )
            score = 0
            if eligible:
                grade_sum, count = compute_grade_sum(olevel_subjects, course, parsed_req)
                course_weight = course_weights.get(course, 0)
                score = compute_enhanced_score(utme_score, grade_sum, count, course_weight)
            
            results.append({
                "course": course, 
                "eligible": eligible, 
                "score": score,
                "interest_weight": course_weights.get(course, 0)
            })

        results_df = pd.DataFrame(results)
        eligible_courses = results_df[results_df["eligible"] & (results_df["score"] > 0)]

        if eligible_courses.empty:
            return {
                "predicted_program": "UNASSIGNED",
                "score": 0,
                "reason": "No eligible course found",
            }

        # Select the course with the highest score
        best_course = eligible_courses.loc[eligible_courses["score"].idxmax()]
        return {
            "predicted_program": best_course["course"],
            "score": best_course["score"],
            "reason": "Best match based on interests and qualifications",
            "all_eligible": eligible_courses.sort_values("score", ascending=False),
            "interest_alignment": best_course["interest_weight"] > 0
        }

    st.sidebar.header("📝 Enter Your Details")

    # UTME Score input
    utme_score = st.sidebar.number_input(
        "UTME Score (0-400)",
        min_value=0,
        max_value=400,
        value=250,
        step=1,
        help="Enter your Unified Tertiary Matriculation Examination score",
    )

    st.sidebar.subheader("🎯 Your Interests")
    selected_interests = st.sidebar.multiselect(
        "Select your areas of interest (choose 2-4):",
        options=list(interest_categories.keys()),
        default=["Problem Solving & Logic"],
        help="Select multiple areas that interest you most"
    )

    learning_style = st.sidebar.selectbox(
        "Learning Style",
        options=list(learning_styles.keys()),
        help="How do you prefer to learn and work?"
    )

    # O'Level subjects and grades
    st.sidebar.subheader("📚 O'Level Results")
    st.sidebar.markdown("*Enter your O'Level subject grades*")

    olevel_subjects = {}
    for i in range(7):  # Increased to 7 subjects for better matching
        col1, col2 = st.sidebar.columns(2)
        with col1:
            subject = st.selectbox(
                f"Subject {i+1}",
                options=[""] + common_subjects,
                key=f"subject_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
        with col2:
            if subject:
                grade = st.selectbox(
                    f"Grade {i+1}",
                    options=list(grade_map.keys()),
                    key=f"grade_{i}",
                    label_visibility="collapsed",
                )
                olevel_subjects[subject] = grade_map[grade]

    # Predict button
    if st.sidebar.button("🔮 Get My Personalized Recommendations", type="primary"):
        if len(olevel_subjects) >= 5 and len(selected_interests) >= 1:  # Minimum requirements
            st.session_state.prediction_made = True
            st.session_state.prediction_result = predict_placement_enhanced(
                utme_score, olevel_subjects, selected_interests, learning_style
            )
            st.session_state.user_data = {
                "utme_score": utme_score,
                "olevel_subjects": olevel_subjects,
                "selected_interests": selected_interests,
                "learning_style": learning_style,
            }
        else:
            st.sidebar.error("Please enter at least 5 O'Level subjects and select at least 1 interest area")

    # ... existing code for displaying individual recommendations ...

    # Main content area
    if st.session_state.prediction_made:
        result = st.session_state.prediction_result
        user_data = st.session_state.user_data

        st.header("🎯 Your Personalized Course Recommendations")

        if result["predicted_program"] != "UNASSIGNED":
            # Success case with enhanced metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="🏆 Top Recommendation",
                    value=result["predicted_program"],
                    help="Your best course match based on comprehensive analysis",
                )

            with col2:
                st.metric(
                    label="📊 Match Score",
                    value=f"{result['score']:.1%}",
                    help="Overall compatibility score",
                )

            with col3:
                st.metric(
                    label="🎯 Interest Alignment",
                    value="✅ High" if result.get("interest_alignment", False) else "⚠️ Moderate",
                    help="How well this course matches your interests",
                )

            with col4:
                st.metric(
                    label="📈 UTME Status",
                    value=f"{user_data['utme_score']}/{cutoff_marks[result['predicted_program']]}",
                    delta=f"{user_data['utme_score'] - cutoff_marks[result['predicted_program']]} above cutoff",
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

            # Enhanced visualization
            if "all_eligible" in result and len(result["all_eligible"]) > 1:
                st.subheader("📊 All Eligible Courses Analysis")

                # Top 10 courses with enhanced visualization
                eligible_df = result["all_eligible"].head(10)
                
                # Create enhanced bar chart
                fig = go.Figure()
                
                colors = ['#1f77b4' if course == result["predicted_program"] else '#aec7e8' 
                         for course in eligible_df["course"]]
                
                fig.add_trace(go.Bar(
                    y=eligible_df["course"],
                    x=eligible_df["score"],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{score:.1%}" for score in eligible_df["score"]],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Top 10 Course Matches with Interest Weighting",
                    xaxis_title="Match Score",
                    yaxis_title="Course",
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Enhanced table with additional information
                display_df = eligible_df.copy()
                display_df["Interest Match"] = display_df["interest_weight"].apply(
                    lambda x: "🎯 High" if x > 0.2 else "⚠️ Moderate" if x > 0 else "❌ Low"
                )
                display_df = display_df[["course", "score", "Interest Match"]].rename(
                    lambda x: "🎯 High" if x > 0.2 else "⚠️ Moderate" if x > 0 else "❌ Low"
                )
                display_df = display_df[["course", "score", "Interest Match"]].rename(
                    columns={"course": "Course", "score": "Match Score"}
                )
                
                st.dataframe(display_df, use_container_width=True)

        else:
            # Enhanced error handling with suggestions
            st.error("❌ No Eligible Course Found")
            st.markdown(f"**Reason:** {result['reason']}")

            st.subheader("💡 Improvement Suggestions")
            suggestions = []
            
            if user_data["utme_score"] < 180:
                suggestions.append("📈 Consider retaking UTME to improve your score (minimum 180 required)")
            
            if len(user_data["olevel_subjects"]) < 5:
                suggestions.append("📚 Ensure you have at least 5 O'Level credits")
                
            suggestions.extend([
                "🔍 Review O'Level requirements for your preferred courses",
                "🎯 Consider alternative courses with lower entry requirements",
                "📞 Consult with academic advisors for personalized guidance"
            ])
            
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")

    else:
        st.header("Welcome to the Individual Course Recommender! 👋")

        # Feature highlights
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 🚀 Advanced Features:
            - **Multi-Interest Matching** - Select multiple areas of interest
            - **Learning Style Analysis** - Personalized based on how you learn
            - **Career Path Information** - See potential career outcomes
            - **Enhanced Scoring Algorithm** - More accurate recommendations
            - **Professional Insights** - Salary expectations and job outlook
            """
            )

        with col2:
            st.markdown(
                """
            ### 📊 How It Works:
            1. **Academic Assessment** - Enter UTME score and O'Level results
            2. **Interest Profiling** - Select your areas of interest
            3. **Learning Style** - Choose your preferred learning approach
            4. **AI Analysis** - Advanced algorithm processes your profile
            5. **Personalized Results** - Get tailored recommendations with career insights
            """
            )

        # Interest categories overview
        st.subheader("🎯 Available Interest Categories")
        
        interest_cols = st.columns(3)
        for i, (category, courses) in enumerate(interest_categories.items()):
            with interest_cols[i % 3]:
                with st.expander(f"📚 {category}"):
                    st.write(f"**Sample Courses:** {', '.join(courses[:3])}")
                    if len(courses) > 3:
                        st.write(f"*...and {len(courses) - 3} more*")

        # System statistics
        st.subheader("📈 System Capabilities")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Available Courses", len(course_names))
        with col2:
            st.metric("Interest Categories", len(interest_categories))
        with col3:
            st.metric("Learning Styles", len(learning_styles))
        with col4:
            st.metric("O'Level Subjects", len(common_subjects))

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <h4>🎓 University Admission Management System</h4>
        <p>Powered by ML algorithms for intelligent course allocation and admission processing</p>
        <p><em>Helping universities make data-driven admission decisions</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)
