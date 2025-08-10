#!/usr/bin/env python3
"""
Unified Student Data Dashboard
Combines basic dashboard, AI analysis, and chatbot into one comprehensive web application.
"""

from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import io
import json
import warnings
# Remove Flask-Mail import and config
# from flask_mail import Mail, Message
warnings.filterwarnings('ignore')

# Try to import AI libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'unified-dashboard-secret-key'

# Configure mail
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your_gmail@gmail.com'
# app.config['MAIL_PASSWORD'] = 'your_app_password'  # Use an app password, not your real password

# mail = Mail(app)

# Global variables
student_data = None
ai_analyzer = None

class UnifiedStudentAnalyzer:
    """Unified analyzer combining all features"""
    
    def __init__(self, df):
        self.df = df
        self.setup_ai_features()
    
    def setup_ai_features(self):
        """Setup AI features for analysis"""
        if not AI_AVAILABLE:
            return
        
        # Create interest-based features
        all_interests = set()
        for interests in self.df['Interests_List']:
            all_interests.update(interests)
        
        # One-hot encoding for interests
        for interest in all_interests:
            self.df[f'AI_Interest_{interest.replace(" ", "_").replace("/", "_")}'] = self.df['Interests_List'].apply(
                lambda x: 1 if interest in x else 0
            )
        
        # Engagement score
        self.df['AI_Engagement_Score'] = (
            self.df['Interest_Count'] * 10 +
            self.df['Valid_Email'] * 20 +
            (self.df['Year_Clean'].isin(['3', '4', '5']) * 15) +
            np.random.normal(50, 10, len(self.df))
        )
        
        # Success potential
        np.random.seed(42)
        self.df['AI_Success_Potential'] = np.random.normal(75, 15, len(self.df))
    
    def perform_clustering(self, n_clusters=5):
        """Perform student clustering"""
        if not AI_AVAILABLE:
            return None
        
        try:
            interest_cols = [col for col in self.df.columns if col.startswith('AI_Interest_')]
            features = self.df[interest_cols + ['Interest_Count']].copy()
            features['Year_Numeric'] = pd.to_numeric(self.df['Year_Clean'], errors='coerce').fillna(0)
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['Cluster'] = kmeans.fit_predict(features_scaled)
            
            cluster_analysis = self.df.groupby('Cluster').agg({
                'Interest_Count': ['mean', 'count'],
                'Branch': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'Year_Clean': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(2)
            
            cluster_analysis.columns = ['Avg_Interests', 'Student_Count', 'Most_Common_Branch', 'Most_Common_Year']
            
            return cluster_analysis.to_dict('index')
        except Exception as e:
            print(f"Clustering error: {e}")
            return None
    
    def generate_recommendations(self):
        """Generate AI recommendations"""
        if not AI_AVAILABLE:
            return []
        
        try:
            interest_cols = [col for col in self.df.columns if col.startswith('AI_Interest_')]
            interest_matrix = self.df[interest_cols].values
            similarity_matrix = cosine_similarity(interest_matrix)
            
            recommendations = {
                'AI/ML': ['Machine Learning Workshop', 'AI Research Lab', 'Data Science Club'],
                'Web Dev': ['Web Development Bootcamp', 'Frontend Workshop', 'Hackathon'],
                'App Dev': ['Mobile App Development', 'React Native Workshop', 'App Store Optimization'],
                'Cybersecurity': ['Security Workshop', 'Ethical Hacking Course', 'CTF Competition'],
                'Blockchain': ['Blockchain Development', 'Smart Contract Workshop', 'Crypto Club'],
                'Cloud': ['AWS Certification', 'Cloud Architecture Workshop', 'DevOps Training'],
                'Robotics': ['Robotics Club', 'IoT Workshop', 'Hardware Hacking'],
                'DSA': ['Competitive Programming', 'Algorithm Workshop', 'Coding Contest']
            }
            
            self.df['Recommendations'] = ''
            
            for idx, student in self.df.iterrows():
                student_recommendations = []
                
                # Get recommendations based on interests
                for interest in student['Interests_List']:
                    if interest in recommendations:
                        student_recommendations.extend(recommendations[interest][:2])
                
                # Remove duplicates and limit
                student_recommendations = list(set(student_recommendations))[:5]
                self.df.at[idx, 'Recommendations'] = '; '.join(student_recommendations)
            
            return self.df['Recommendations'].tolist()
        except Exception as e:
            print(f"Recommendation error: {e}")
            return []
    
    def predict_success(self):
        """Predict student success"""
        if not AI_AVAILABLE:
            return None
        
        try:
            feature_cols = ['Interest_Count'] + [col for col in self.df.columns if col.startswith('AI_Interest_')]
            self.df['Year_Numeric'] = pd.to_numeric(self.df['Year_Clean'], errors='coerce').fillna(0)
            feature_cols.append('Year_Numeric')
            
            X = self.df[feature_cols].fillna(0)
            
            # Create simulated success target
            success_threshold = self.df['AI_Success_Potential'].mean()
            y = (self.df['AI_Success_Potential'] > success_threshold).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Add predictions
            self.df['Predicted_Success'] = rf_model.predict(X)
            self.df['Success_Probability'] = rf_model.predict_proba(X)[:, 1]
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return {
                'high_potential': self.df[self.df['Success_Probability'] > 0.8].head(10).to_dict('records'),
                'feature_importance': feature_importance.head(10).to_dict('records')
            }
        except Exception as e:
            print(f"Success prediction error: {e}")
            return None

def load_and_clean_data():
    """Load and clean student data"""
    global student_data, ai_analyzer
    
    try:
        df = pd.read_csv('student_data_with_edge_cases.xlsx - Sheet1.csv')
        df = df.fillna('')
        
        # Clean Year column
        def standardize_year(year):
            if pd.isna(year) or year == '':
                return 'Unknown'
            year_str = str(year).strip().lower()
            if year_str == 'second':
                return '2'
            try:
                return str(int(float(year_str)))
            except:
                return year_str
        
        df['Year_Clean'] = df['Year'].apply(standardize_year)
        
        # Clean Email column
        def is_valid_email(email):
            if pd.isna(email) or email == '':
                return False
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, str(email)))
        
        df['Valid_Email'] = df['Email'].apply(is_valid_email)
        
        # Clean Interests
        def clean_interests(interests):
            if pd.isna(interests) or interests == '':
                return []
            interest_list = [interest.strip() for interest in str(interests).split(',')]
            return [interest for interest in interest_list if interest]
        
        df['Interests_List'] = df['Interests'].apply(clean_interests)
        df['Interest_Count'] = df['Interests_List'].apply(len)
        df['Student_ID'] = range(1, len(df) + 1)
        
        student_data = df
        ai_analyzer = UnifiedStudentAnalyzer(df)
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    if student_data is None:
        load_and_clean_data()
    
    # Get unique values for filters
    branches = sorted(student_data['Branch'].unique().tolist())
    years = sorted(student_data['Year_Clean'].unique().tolist())
    interests = []
    for interest_list in student_data['Interests_List']:
        if interest_list and isinstance(interest_list, list):
            interests.extend(interest_list)
    interests = sorted(list(set(interests)))
    
    return render_template('unified_dashboard.html', 
                         branches=branches, 
                         years=years, 
                         interests=interests,
                         ai_available=AI_AVAILABLE)

@app.route('/api/students')
def get_students():
    """API endpoint to get filtered student data"""
    if student_data is None:
        load_and_clean_data()
    
    # Get filter parameters
    branch_filter = request.args.get('branch', '')
    year_filter = request.args.get('year', '')
    interest_filter = request.args.get('interest', '')
    search_query = request.args.get('search', '').lower()
    chinese_only = request.args.get('chinese', 'false').lower() == 'true'  # NEW

    # Apply filters
    filtered_data = student_data.copy()
    
    if branch_filter:
        filtered_data = filtered_data[filtered_data['Branch'] == branch_filter]
    
    if year_filter:
        if year_filter == "2":
            filtered_data = filtered_data[
                (filtered_data['Year_Clean'] == '2') |
                (filtered_data['Year'].astype(str).str.contains('second', case=False, na=False))
            ]
        else:
            filtered_data = filtered_data[filtered_data['Year_Clean'] == year_filter]
    
    if interest_filter:
        filtered_data = filtered_data[filtered_data['Interests_List'].apply(
            lambda x: any(interest_filter.lower() in str(i).lower() for i in x)
        )]
    
    if search_query:
        mask = (
            filtered_data['Name'].str.lower().str.contains(search_query, na=False) |
            filtered_data['Email'].str.lower().str.contains(search_query, na=False) |
            filtered_data['Branch'].str.lower().str.contains(search_query, na=False) |
            filtered_data['Interests'].str.lower().str.contains(search_query, na=False)
        )
        filtered_data = filtered_data[mask]

    # --- Filter for Chinese names ---
    if chinese_only:
        import re
        filtered_data = filtered_data[
            filtered_data['Name'].apply(lambda x: bool(re.search(r'[\u4e00-\u9fff]', str(x))))
            | filtered_data['Interests'].apply(lambda x: bool(re.search(r'[\u4e00-\u9fff]', str(x))))
        ]
    # ---

    # Convert to list of dictionaries
    students = filtered_data.to_dict('records')
    
    return jsonify({
        'students': students,
        'total_count': len(student_data),
        'filtered_count': len(filtered_data)
    })

@app.route('/api/ai/clustering')
def ai_clustering():
    """AI clustering endpoint"""
    if not AI_AVAILABLE:
        return jsonify({'error': 'AI features not available'})
    
    try:
        cluster_analysis = ai_analyzer.perform_clustering()
        return jsonify({'success': True, 'clusters': cluster_analysis})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ai/recommendations')
def ai_recommendations():
    """AI recommendations endpoint"""
    if not AI_AVAILABLE:
        return jsonify({'error': 'AI features not available'})
    
    try:
        recommendations = ai_analyzer.generate_recommendations()
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ai/success-prediction')
def ai_success_prediction():
    """AI success prediction endpoint"""
    if not AI_AVAILABLE:
        return jsonify({'error': 'AI features not available'})
    
    try:
        prediction_results = ai_analyzer.predict_success()
        return jsonify({'success': True, 'predictions': prediction_results})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Chatbot endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Simple chatbot logic
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = "Hello! I'm your AI student data assistant. How can I help you today?"
        elif 'statistics' in query_lower or 'stats' in query_lower:
            total = len(student_data)
            valid_emails = student_data['Valid_Email'].sum()
            response = f"📊 Student Statistics:\n• Total Students: {total}\n• Valid Emails: {valid_emails}\n• Average Interests: {student_data['Interest_Count'].mean():.1f}"
        elif 'cse' in query_lower:
            cse_count = len(student_data[student_data['Branch'] == 'CSE'])
            response = f"📊 There are {cse_count} CSE students."
        elif 'ai/ml' in query_lower or 'ai' in query_lower:
            ai_students = student_data[student_data['Interests_List'].apply(
                lambda x: any('ai' in str(i).lower() for i in x)
            )]
            response = f"🎯 Found {len(ai_students)} students interested in AI/ML."
        elif 'year 2' in query_lower or 'second' in query_lower:
            year2_students = student_data[
                (student_data['Year_Clean'] == '2') |
                (student_data['Year'].astype(str).str.contains('second', case=False, na=False))
            ]
            response = f"📅 Found {len(year2_students)} students in Year 2 (including 'second')."
        else:
            response = "I understand you're asking about student data. Try asking about statistics, specific branches, interests, or years!"
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/export-csv')
def export_csv():
    """Export filtered data as CSV"""
    try:
        # Get current filters from request
        branch_filter = request.args.get('branch', '')
        year_filter = request.args.get('year', '')
        interest_filter = request.args.get('interest', '')
        search_query = request.args.get('search', '').lower()
        
        
        # Apply same filters as in get_students
        filtered_data = student_data.copy()
        
        if branch_filter:
            filtered_data = filtered_data[filtered_data['Branch'] == branch_filter]
        
        if year_filter:
            if year_filter == "2":
                filtered_data = filtered_data[
                    (filtered_data['Year_Clean'] == '2') |
                    (filtered_data['Year'].astype(str).str.contains('second', case=False, na=False))
                ]
            else:
                filtered_data = filtered_data[filtered_data['Year_Clean'] == year_filter]
        
        if interest_filter:
            filtered_data = filtered_data[filtered_data['Interests_List'].apply(
                lambda x: any(interest_filter.lower() in str(i).lower() for i in x)
            )]
        
        if search_query:
            mask = (
                filtered_data['Name'].str.lower().str.contains(search_query, na=False) |
                filtered_data['Email'].str.lower().str.contains(search_query, na=False) |
                filtered_data['Branch'].str.lower().str.contains(search_query, na=False) |
                filtered_data['Interests'].str.lower().str.contains(search_query, na=False)
            )
            filtered_data = filtered_data[mask]
        
        # Remove AI columns for export
        export_cols = ['Name', 'Branch', 'Year', 'Email', 'Interests']
        export_data = filtered_data[export_cols]
        
        # Create CSV in memory
        output = io.StringIO()
        export_data.to_csv(output, index=False)
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'unified_student_export_{timestamp}.csv'
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error exporting CSV: {str(e)}'})

@app.route('/api/upload-excel', methods=['POST'])
def upload_excel():
    """Upload and process Excel or CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
            # Save file temporarily
            if file.filename.endswith('.xlsx'):
                file.save('student_data_with_edge_cases.xlsx')
            else:
                file.save('student_data_with_edge_cases.csv')
            
            # Reload data
            load_and_clean_data()
            
            return jsonify({
                'success': True, 
                'message': 'File uploaded and processed successfully',
                'total_students': len(student_data)
            })
        else:
            return jsonify({'success': False, 'message': 'Please upload an Excel (.xlsx) or CSV (.csv) file'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})

@app.route('/api/send-emails', methods=['POST'])
def send_emails():
    """Simulate sending emails to students (no actual email sent)"""
    try:
        data = request.get_json()
        emails = data.get('emails', [])
        subject = data.get('subject', 'Personalized Message')
        message = data.get('message', 'Hello from the dashboard!')
        if not emails:
            return jsonify({'success': False, 'message': 'No emails provided'})
        # Simulate sending by printing/logging
        print(f"Simulated sending email to: {emails} | Subject: {subject} | Message: {message}")
        return jsonify({'success': True, 'message': 'Emails simulated (not actually sent).'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing email: {str(e)}'})

if __name__ == '__main__':
    # Load data on startup
    load_and_clean_data()
    print("🚀 Unified Student Dashboard starting...")
    print("📊 Basic Dashboard: http://localhost:5000")
    print("🤖 AI Features: Available" if AI_AVAILABLE else "🤖 AI Features: Not available")
    print("💬 Chatbot: Integrated")
    app.run(debug=True, host='0.0.0.0', port=5000)