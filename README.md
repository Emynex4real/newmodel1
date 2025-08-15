# 🎓 Intelligent Course Recommender System

A sophisticated Streamlit web application that predicts the best course matches for university applicants based on UTME scores, O'Level results, and course preferences.

## Features

- **Smart Matching Algorithm**: Uses advanced scoring system combining UTME scores, O'Level grades, and interest matching
- **Visual Analytics**: Interactive charts and visualizations using Plotly
- **Course Eligibility Checking**: Validates against course-specific requirements and cutoff marks
- **Personalized Recommendations**: Tailored suggestions based on individual qualifications
- **Alternative Course Suggestions**: Shows backup options when primary choice isn't available

## Quick Start

### Local Development

1. **Clone or download this repository**

2. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Run the application**:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

4. **Open your browser** to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Set main file path**: `app.py`
5. **Deploy!**

## How to Use

1. **Enter your UTME score** (0-400 range)
2. **Select your preferred course** from the dropdown
3. **Input your O'Level results** (minimum 3 subjects required)
4. **Click "Predict My Best Course"** to get recommendations

## System Requirements

- Python 3.7+
- Streamlit 1.28.0+
- See `requirements.txt` for full dependency list

## Course Categories

The system covers 49 courses across 8 major categories:
- Agriculture (9 courses)
- Engineering (8 courses) 
- Science (14 courses)
- Technology (4 courses)
- Health (3 courses)
- Management (5 courses)
- Design (2 courses)
- Geoscience (1 course)

## Algorithm Overview

The recommendation system uses a multi-factor scoring approach:

1. **Eligibility Check**: Validates O'Level requirements and UTME cutoff marks
2. **Score Calculation**: 
   - 50% UTME score (normalized)
   - 50% O'Level grades (normalized, lower grades = better)
   - 10% bonus for interest/related course match
3. **Ranking**: Courses ranked by composite score for personalized recommendations

## Files Structure

\`\`\`
├── app.py                    # Main Streamlit application
├── intelligent_model.ipynb  # Model development and testing
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .streamlit/
    └── config.toml          # Streamlit configuration
\`\`\`

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## License

This project is open source and available under the MIT License.
