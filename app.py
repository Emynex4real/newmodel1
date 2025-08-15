import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Course Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def safe_load_data():
    """Safely load data with error handling"""
    try:
        return load_course_data()
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return None, None, None, None, None

# Title and description
st.title("🎓 Intelligent Course Recommender System")
st.markdown(
    "**Predict your best course match based on UTME scores and O'Level results**"
)

# Initialize session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

try:
    data_result = safe_load_data()
    if data_result[0] is not None:
        common_subjects, grade_map, course_names, course_groups, cutoff_marks = data_result
    else:
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize application: {str(e)}")
    st.stop()

# Define constants and data structures
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

    # Course groups for interest matching
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

    return common_subjects, grade_map, course_names, course_groups, cutoff_marks


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


def get_related_courses(interest, course_groups):
    """Get related courses based on applicant's interest"""
    for group, courses in course_groups.items():
        if interest in courses:
            return courses
    return []


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


def compute_score(utme_score, grade_sum, count, interest_match, interest_bonus=0.1):
    """Compute applicant score for a course"""
    if count == 0:
        return 0

    normalized_utme = utme_score / 400  # Normalize UTME score (0-400)
    average_grade = grade_sum / count
    normalized_grade = (9 - average_grade) / 8  # Normalize grades (lower is better)
    score = 0.5 * normalized_utme + 0.5 * normalized_grade

    if interest_match:
        score += interest_bonus

    return max(0, min(score, 1 + interest_bonus))


def predict_placement(utme_score, olevel_subjects, top_interest):
    """Predict the program an applicant is most likely to be assigned to"""
    parsed_req = get_course_requirements()

    if not isinstance(utme_score, (int, float)) or utme_score < 0 or utme_score > 400:
        return {
            "predicted_program": "UNASSIGNED",
            "score": 0,
            "reason": "Invalid UTME score",
        }

    if top_interest not in course_names:
        return {
            "predicted_program": "UNASSIGNED",
            "score": 0,
            "reason": "Invalid course interest",
        }

    results = []
    related = get_related_courses(top_interest, course_groups)

    for course in course_names:
        eligible = (
            is_eligible(olevel_subjects, course, parsed_req)
            and utme_score >= cutoff_marks[course]
        )
        score = 0
        if eligible:
            grade_sum, count = compute_grade_sum(olevel_subjects, course, parsed_req)
            interest_match = course == top_interest or course in related
            score = compute_score(utme_score, grade_sum, count, interest_match)
        results.append({"course": course, "eligible": eligible, "score": score})

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
        "reason": (
            "Highest score match"
            if best_course["course"] == top_interest
            else "Best alternative match"
        ),
        "all_eligible": eligible_courses.sort_values("score", ascending=False),
    }


# Sidebar for input
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

# Top interest selection
top_interest = st.sidebar.selectbox(
    "Preferred Course",
    options=course_names,
    index=(
        course_names.index("Computer Science")
        if "Computer Science" in course_names
        else 0
    ),
    help="Select your most preferred course",
)

# O'Level subjects and grades
st.sidebar.subheader("O'Level Results")
st.sidebar.markdown("*Enter your O'Level subject grades*")

olevel_subjects = {}
for i in range(5):
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
if st.sidebar.button("🔮 Predict My Best Course", type="primary"):
    if len(olevel_subjects) >= 3:  # Minimum 3 subjects required
        st.session_state.prediction_made = True
        st.session_state.prediction_result = predict_placement(
            utme_score, olevel_subjects, top_interest
        )
        st.session_state.user_data = {
            "utme_score": utme_score,
            "olevel_subjects": olevel_subjects,
            "top_interest": top_interest,
        }
    else:
        st.sidebar.error("Please enter at least 3 O'Level subjects")

# Main content area
if st.session_state.prediction_made:
    result = st.session_state.prediction_result
    user_data = st.session_state.user_data

    # Display prediction results
    st.header("🎯 Your Course Recommendation")

    if result["predicted_program"] != "UNASSIGNED":
        # Success case
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Recommended Course",
                value=result["predicted_program"],
                help="This is your best course match based on your qualifications",
            )

        with col2:
            st.metric(
                label="Match Score",
                value=f"{result['score']:.2%}",
                help="Higher scores indicate better matches",
            )

        with col3:
            st.metric(
                label="UTME Cutoff",
                value=f"{cutoff_marks[result['predicted_program']]}",
                delta=(
                    f"{user_data['utme_score'] - cutoff_marks[result['predicted_program']]} above cutoff"
                    if user_data["utme_score"]
                    >= cutoff_marks[result["predicted_program"]]
                    else f"{cutoff_marks[result['predicted_program']] - user_data['utme_score']} below cutoff"
                ),
            )

        # Reason for recommendation
        st.info(f"**Recommendation Reason:** {result['reason']}")

        # Show all eligible courses
        if "all_eligible" in result and len(result["all_eligible"]) > 1:
            st.subheader("📊 All Eligible Courses")

            # Create a bar chart of eligible courses
            eligible_df = result["all_eligible"].head(10)  # Top 10 courses

            fig = px.bar(
                eligible_df,
                x="score",
                y="course",
                orientation="h",
                title="Top 10 Course Matches",
                labels={"score": "Match Score", "course": "Course"},
                color="score",
                color_continuous_scale="viridis",
            )
            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            # Display table
            st.dataframe(
                eligible_df[["course", "score"]].rename(
                    columns={"course": "Course", "score": "Match Score"}
                ),
                use_container_width=True,
            )

    else:
        # No eligible course found
        st.error("❌ No Eligible Course Found")
        st.markdown(f"**Reason:** {result['reason']}")

        # Suggestions for improvement
        st.subheader("💡 Suggestions")
        if user_data["utme_score"] < 180:
            st.markdown(
                "- Consider retaking UTME to improve your score (minimum 180 required)"
            )

        st.markdown("- Review O'Level requirements for your preferred courses")
        st.markdown("- Consider alternative courses with lower entry requirements")

    # Course requirements section
    st.subheader("📋 Course Requirements")
    selected_course = st.selectbox(
        "View requirements for:",
        options=course_names,
        index=(
            course_names.index(result["predicted_program"])
            if result["predicted_program"] in course_names
            else 0
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("UTME Cutoff", cutoff_marks[selected_course])
    with col2:
        user_meets_cutoff = user_data["utme_score"] >= cutoff_marks[selected_course]
        st.metric(
            "Your UTME Status",
            "✅ Meets Cutoff" if user_meets_cutoff else "❌ Below Cutoff",
            delta=f"{user_data['utme_score'] - cutoff_marks[selected_course]}",
        )

else:
    # Welcome screen
    st.header("Welcome to the Course Recommender System! 👋")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### How it works:
        1. **Enter your UTME score** (0-400 range)
        2. **Select your preferred course** from available options
        3. **Input your O'Level results** (minimum 3 subjects required)
        4. **Get personalized recommendations** based on:
           - Course eligibility requirements
           - Your academic performance
           - Interest matching
        """
        )

    with col2:
        st.markdown(
            """
        ### Features:
        - ✅ **Smart Matching Algorithm** - Uses advanced scoring system
        - 📊 **Visual Analytics** - See all your eligible courses
        - 🎯 **Personalized Results** - Tailored to your qualifications
        - 📋 **Course Requirements** - View detailed entry requirements
        - 🔄 **Alternative Suggestions** - Find backup options
        """
        )

    # Sample statistics
    st.subheader("📈 System Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Available Courses", len(course_names))
    with col2:
        st.metric("Subject Areas", len(course_groups))
    with col3:
        st.metric("O'Level Subjects", len(common_subjects))
    with col4:
        st.metric("Grade Levels", len(grade_map))

    # Course distribution chart
    st.subheader("🎓 Course Distribution by Category")

    category_counts = {}
    for category, courses in course_groups.items():
        category_counts[category.title()] = len(courses)

    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Available Courses by Category",
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 Intelligent Course Recommender</p>
        <p>Helping students make informed decisions about their academic future</p>
    </div>
    """,
    unsafe_allow_html=True,
)
