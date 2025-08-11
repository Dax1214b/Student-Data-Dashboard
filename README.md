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
  
  • AI-Powered Insights<br>
    • Clustering based on interests.<br>
    • Personalized activity recommendations.<br>
    • Predicts students with high success potential.<br>
  
  • Interactive Chatbot<br>
    • Ask natural language questions like:<br>
      • "Show students interested in AI/ML"<br>
      • "How many CSE students are there?"<br>
      • "List third-year students"<br>
    
  • Data Cleaning & Processing<br>
    • Standardizes year format.<br>
    • Validates email addresses.<br>
    • Extracts and counts interests.<br>
  
  • Visualization<br>
    • Graphs and charts using Matplotlib and Seaborn.<br>

🛠 Tech Stack<br>
  • Backend: Python, Flask<br>
  • Frontend: HTML, CSS, JavaScript<br>
  • Data Processing: Pandas, NumPy<br>
  • Machine Learning: scikit-learn (KMeans, RandomForestClassifier)<br>
  • Visualization: Matplotlib, Seaborn<br>

📂 Project Structure<br>

  ├── ai_student_analysis.py        # AI-based student data analysis<br>
  ├── genai_student_analyzer.py     # General AI analyzer module<br>
  ├── student_analysis.py           # Core analysis logic<br>
  ├── student_chatbot.py            # Chatbot logic for CLI interaction<br>
  ├── unified_dashboard.py          # Main Flask application<br>
  ├── student_data_with_edge_cases.xlsx - Sheet1.csv  # Sample dataset<br>
  ├── student_analysis_dashboard.png # Dashboard screenshot<br>
  ├── student_clusters.png           # Cluster visualization<br>
  └── README.md                      # Project documentation<br>

▶️ How to Open & Run the Project<br>
Option 1 - Run as a Flask Web App<br>
  1. Make sure you are in the project folder and your virtual environment is activated.<br>
    
  2. Run:<br>
      • python unified_dashboard.py<br>
      
  3. Open your browser and go to:<br>
      • 127.0.0.1:5000<br>
      
  4. You’ll see the interactive dashboard where you can:<br>
      • Filter and search students<br>
      • View AI-powered recommendations<br>
      • See success predictions and clusters<br>

Option 2 – Run the Chatbot in Terminal<br>
  1. Run:<br>
      python student_chatbot.py<br>
  2. Type your queries, for example:<br>
      • "Show students interested in AI/ML"<br>
      • "List all second-year students"<br>
      • "How many students are in CSE?"<br>
