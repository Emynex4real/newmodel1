import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json

# Page configuration
st.set_page_config(
    page_title="Advanced Course Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

@st.cache_data
def safe_load_data():
    """Safely load data with error handling"""
    try:
        return load_course_data()
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return None, None, None, None, None, None, None, None

# Title and description
st.title("🎓 Advanced Intelligent Course Recommender System")
st.markdown(
    "**Get personalized course recommendations based on your interests, learning style, and academic performance**"
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
    st.header("Welcome to the Advanced Course Recommender System! 👋")

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
        <h4>🎓 Advanced Intelligent Course Recommender</h4>
        <p>Powered by AI-driven matching algorithms and comprehensive career data</p>
        <p><em>Helping students make informed decisions about their academic and professional future</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)
