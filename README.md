🎓 AI-Powered Student Data Analysis Dashboard
<br>

📌 Overview
<br>
  This project is a Flask-based web application that analyzes and visualizes student data using machine learning.
  It combines:
  
    📊 Interactive Dashboard for filtering, searching, and exploring student data.
    🤖 AI Features for clustering, recommendations, and success prediction.
    💬 Chatbot for natural language queries.

🚀 Features
<br>
  •   Student Filtering & Search<br>
    • Filter by branch, year, interest, or keywords.<br>
    • Special filter for Chinese names.<br>
  
  • AI-Powered Insights
    • Clustering based on interests.
    • Personalized activity recommendations.
    • Predicts students with high success potential.
  
  • Interactive Chatbot
    • Ask natural language questions like:
      • "Show students interested in AI/ML"
      • "How many CSE students are there?"
      • "List third-year students"
    
  • Data Cleaning & Processing
    • Standardizes year format.
    • Validates email addresses.
    • Extracts and counts interests.
  
  • Visualization
    • Graphs and charts using Matplotlib and Seaborn.

🛠 Tech Stack
  • Backend: Python, Flask
  • Frontend: HTML, CSS, JavaScript
  • Data Processing: Pandas, NumPy
  • Machine Learning: scikit-learn (KMeans, RandomForestClassifier)
  • Visualization: Matplotlib, Seaborn

📂 Project Structure

  ├── ai_student_analysis.py        # AI-based student data analysis
  ├── genai_student_analyzer.py     # General AI analyzer module
  ├── student_analysis.py           # Core analysis logic
  ├── student_chatbot.py            # Chatbot logic for CLI interaction
  ├── unified_dashboard.py          # Main Flask application
  ├── student_data_with_edge_cases.xlsx - Sheet1.csv  # Sample dataset
  ├── student_analysis_dashboard.png # Dashboard screenshot
  ├── student_clusters.png           # Cluster visualization
  └── README.md                      # Project documentation

▶️ How to Open & Run the Project
Option 1 - Run as a Flask Web App
  1. Make sure you are in the project folder and your virtual environment is activated.
    
  2. Run:
      • python unified_dashboard.py
      
  3. Open your browser and go to:
      • 127.0.0.1:5000
      
  4. You’ll see the interactive dashboard where you can:
      • Filter and search students
      • View AI-powered recommendations
      • See success predictions and clusters

Option 2 – Run the Chatbot in Terminal
  1. Run:
      python student_chatbot.py
  2. Type your queries, for example:
      • "Show students interested in AI/ML"
      • "List all second-year students"
      • "How many students are in CSE?"
