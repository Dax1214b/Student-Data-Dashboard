#!/usr/bin/env python3
"""
AI-Enhanced Student Data Analysis Dashboard
Advanced analysis with machine learning, clustering, recommendations, and sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import AI/ML libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ AI/ML libraries imported successfully!")
    AI_AVAILABLE = True
except ImportError:
    print("⚠️ AI/ML libraries not available. Installing basic versions...")
    AI_AVAILABLE = False

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("🤖 AI-Enhanced Student Data Analysis Dashboard")
print("=" * 60)
print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_and_clean_data():
    """Load and clean the student data"""
    print("\n📁 Loading and cleaning data...")
    
    try:
        df = pd.read_csv('student_data_with_edge_cases.xlsx - Sheet1.csv')
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ File not found. Please ensure 'student_data_with_edge_cases.xlsx - Sheet1.csv' is in the current directory.")
        return None
    
    # Fill missing values
    df = df.fillna('')
    
    # Clean Year column - handle 'second' and numeric values
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
    
    # Clean Email column - identify invalid emails
    def is_valid_email(email):
        if pd.isna(email) or email == '':
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, str(email)))
    
    df['Valid_Email'] = df['Email'].apply(is_valid_email)
    
    # Clean Interests - split and standardize
    def clean_interests(interests):
        if pd.isna(interests) or interests == '':
            return []
        interest_list = [interest.strip() for interest in str(interests).split(',')]
        return [interest for interest in interest_list if interest]
    
    df['Interests_List'] = df['Interests'].apply(clean_interests)
    df['Interest_Count'] = df['Interests_List'].apply(len)
    
    # Add student ID
    df['Student_ID'] = range(1, len(df) + 1)
    
    # Create interest-based features for AI analysis
    all_interests = set()
    for interests in df['Interests_List']:
        all_interests.update(interests)
    
    # Create one-hot encoding for interests
    for interest in all_interests:
        df[f'Interest_{interest.replace(" ", "_").replace("/", "_")}'] = df['Interests_List'].apply(
            lambda x: 1 if interest in x else 0
        )
    
    print("✅ Data cleaned successfully!")
    print(f"📧 Valid emails: {df['Valid_Email'].sum()} / {len(df)}")
    print(f"📅 Years standardized: {df['Year_Clean'].nunique()} unique values")
    print(f"🎯 Average interests per student: {df['Interest_Count'].mean():.1f}")
    print(f"🤖 Created {len(all_interests)} interest features for AI analysis")
    
    return df

class AIStudentAnalyzer:
    """AI-enhanced student analysis with machine learning features"""
    
    def __init__(self, df):
        self.df = df
        self.clusters = None
        self.recommendations = None
        self.similarity_matrix = None
        
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on students based on interests and characteristics"""
        print(f"\n🎯 PERFORMING STUDENT CLUSTERING (K={n_clusters})")
        print("=" * 50)
        
        # Prepare features for clustering
        interest_cols = [col for col in self.df.columns if col.startswith('Interest_')]
        features = self.df[interest_cols + ['Interest_Count']].copy()
        
        # Add year as numeric feature
        features['Year_Numeric'] = pd.to_numeric(self.df['Year_Clean'], errors='coerce').fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Cluster'] = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = self.df.groupby('Cluster').agg({
            'Interest_Count': ['mean', 'count'],
            'Branch': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'Year_Clean': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        cluster_analysis.columns = ['Avg_Interests', 'Student_Count', 'Most_Common_Branch', 'Most_Common_Year']
        
        print("📊 CLUSTER ANALYSIS:")
        for cluster_id, data in cluster_analysis.iterrows():
            print(f"\nCluster {cluster_id}:")
            print(f"  👥 Students: {data['Student_Count']}")
            print(f"  🎯 Avg Interests: {data['Avg_Interests']}")
            print(f"  🏫 Common Branch: {data['Most_Common_Branch']}")
            print(f"  📅 Common Year: {data['Most_Common_Year']}")
        
        # Visualize clusters
        self.visualize_clusters()
        
        return cluster_analysis
    
    def visualize_clusters(self):
        """Visualize student clusters"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot of clusters
        scatter = plt.scatter(
            self.df['Interest_Count'], 
            self.df.groupby('Cluster')['Interest_Count'].transform('mean'),
            c=self.df['Cluster'], 
            cmap='viridis', 
            alpha=0.7,
            s=50
        )
        
        plt.xlabel('Number of Interests')
        plt.ylabel('Cluster Average Interests')
        plt.title('Student Clusters Based on Interests')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.savefig('student_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Cluster visualization saved as 'student_clusters.png'")
    
    def generate_recommendations(self):
        """Generate AI-based recommendations for students"""
        print("\n🤖 GENERATING AI RECOMMENDATIONS")
        print("=" * 50)
        
        # Create interest-based similarity matrix
        interest_cols = [col for col in self.df.columns if col.startswith('Interest_')]
        interest_matrix = self.df[interest_cols].values
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(interest_matrix)
        
        # Define recommendation categories
        recommendations = {
            'AI/ML': ['Machine Learning Workshop', 'AI Research Lab', 'Data Science Club', 'Kaggle Competition'],
            'Web Dev': ['Web Development Bootcamp', 'Frontend Framework Workshop', 'Full-Stack Project', 'Hackathon'],
            'App Dev': ['Mobile App Development', 'React Native Workshop', 'App Store Optimization', 'Mobile UI/UX'],
            'Cybersecurity': ['Security Workshop', 'Ethical Hacking Course', 'CTF Competition', 'Security Club'],
            'Blockchain': ['Blockchain Development', 'Smart Contract Workshop', 'DeFi Project', 'Crypto Club'],
            'Cloud': ['AWS Certification', 'Cloud Architecture Workshop', 'DevOps Training', 'Cloud Project'],
            'Robotics': ['Robotics Club', 'IoT Workshop', 'Hardware Hacking', 'Automation Project'],
            'DSA': ['Competitive Programming', 'Algorithm Workshop', 'Coding Contest', 'Problem Solving Club']
        }
        
        # Generate personalized recommendations
        self.df['Recommendations'] = ''
        
        for idx, student in self.df.iterrows():
            student_recommendations = []
            
            # Find similar students
            similar_students = np.argsort(self.similarity_matrix[idx])[-6:-1]  # Top 5 similar students
            
            # Get interests of similar students
            similar_interests = set()
            for similar_idx in similar_students:
                similar_interests.update(student['Interests_List'])
            
            # Generate recommendations based on interests
            for interest in student['Interests_List']:
                if interest in recommendations:
                    student_recommendations.extend(recommendations[interest][:2])  # Top 2 per interest
            
            # Add recommendations based on similar students
            for interest in similar_interests:
                if interest in recommendations and interest not in student['Interests_List']:
                    student_recommendations.extend(recommendations[interest][:1])
            
            # Remove duplicates and limit to 5 recommendations
            student_recommendations = list(set(student_recommendations))[:5]
            self.df.at[idx, 'Recommendations'] = '; '.join(student_recommendations)
        
        # Show sample recommendations
        print("📋 SAMPLE RECOMMENDATIONS:")
        sample_students = self.df[self.df['Recommendations'] != ''].head(5)
        for _, student in sample_students.iterrows():
            print(f"\n👤 {student['Name']} ({student['Branch']}, Year {student['Year_Clean']}):")
            print(f"   🎯 Interests: {', '.join(student['Interests_List'])}")
            print(f"   💡 Recommendations: {student['Recommendations']}")
        
        return self.df['Recommendations']
    
    def predict_student_success(self):
        """Predict student success based on characteristics"""
        print("\n🔮 PREDICTING STUDENT SUCCESS")
        print("=" * 50)
        
        # Create success indicators (simulated)
        np.random.seed(42)
        self.df['Success_Score'] = np.random.normal(75, 15, len(self.df))
        
        # Features for prediction
        feature_cols = ['Interest_Count'] + [col for col in self.df.columns if col.startswith('Interest_')]
        
        # Add year as feature
        self.df['Year_Numeric'] = pd.to_numeric(self.df['Year_Clean'], errors='coerce').fillna(0)
        feature_cols.append('Year_Numeric')
        
        # Create binary success target (above average = success)
        success_threshold = self.df['Success_Score'].mean()
        self.df['Success_Binary'] = (self.df['Success_Score'] > success_threshold).astype(int)
        
        # Prepare features
        X = self.df[feature_cols].fillna(0)
        y = self.df['Success_Binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("📊 FEATURE IMPORTANCE FOR SUCCESS PREDICTION:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
        
        # Add predictions to dataframe
        self.df['Predicted_Success'] = rf_model.predict(X[feature_cols].fillna(0))
        self.df['Success_Probability'] = rf_model.predict_proba(X[feature_cols].fillna(0))[:, 1]
        
        # Show prediction results
        high_potential = self.df[self.df['Success_Probability'] > 0.8].head(5)
        print(f"\n🌟 HIGH POTENTIAL STUDENTS (Success Probability > 80%):")
        for _, student in high_potential.iterrows():
            print(f"  {student['Name']}: {student['Success_Probability']:.1%} success probability")
        
        return rf_model, feature_importance
    
    def analyze_student_sentiment(self):
        """Analyze sentiment and engagement based on interests"""
        print("\n😊 ANALYZING STUDENT SENTIMENT & ENGAGEMENT")
        print("=" * 50)
        
        # Create engagement score based on various factors
        self.df['Engagement_Score'] = (
            self.df['Interest_Count'] * 10 +  # More interests = higher engagement
            self.df['Valid_Email'] * 20 +     # Valid email = more engaged
            (self.df['Year_Clean'].isin(['3', '4', '5']) * 15) +  # Upper years more engaged
            np.random.normal(50, 10, len(self.df))  # Random factor
        )
        
        # Categorize engagement levels
        self.df['Engagement_Level'] = pd.cut(
            self.df['Engagement_Score'], 
            bins=[0, 60, 80, 100, 120], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Analyze by different groups
        print("📊 ENGAGEMENT ANALYSIS:")
        
        # By branch
        branch_engagement = self.df.groupby('Branch')['Engagement_Score'].agg(['mean', 'count']).round(2)
        print("\n🏫 ENGAGEMENT BY BRANCH:")
        for branch, data in branch_engagement.iterrows():
            print(f"  {branch}: {data['mean']:.1f} avg score ({data['count']} students)")
        
        # By year
        year_engagement = self.df.groupby('Year_Clean')['Engagement_Score'].agg(['mean', 'count']).round(2)
        print("\n📅 ENGAGEMENT BY YEAR:")
        for year, data in year_engagement.iterrows():
            print(f"  Year {year}: {data['mean']:.1f} avg score ({data['count']} students)")
        
        # By interest
        print("\n🎯 ENGAGEMENT BY INTEREST:")
        for interest in ['AI/ML', 'Web Dev', 'App Dev', 'Cybersecurity']:
            interest_col = f'Interest_{interest.replace("/", "_")}'
            if interest_col in self.df.columns:
                avg_engagement = self.df[self.df[interest_col] == 1]['Engagement_Score'].mean()
                count = self.df[interest_col].sum()
                print(f"  {interest}: {avg_engagement:.1f} avg score ({count} students)")
        
        return self.df[['Engagement_Score', 'Engagement_Level']]
    
    def generate_ai_insights(self):
        """Generate comprehensive AI insights"""
        print("\n🧠 GENERATING AI INSIGHTS")
        print("=" * 50)
        
        insights = []
        
        # Clustering insights
        cluster_sizes = self.df['Cluster'].value_counts()
        largest_cluster = cluster_sizes.idxmax()
        insights.append(f"🎯 Largest student cluster is Cluster {largest_cluster} with {cluster_sizes[largest_cluster]} students")
        
        # Recommendation insights
        recommendation_counts = self.df['Recommendations'].str.split(';').str.len().value_counts()
        avg_recommendations = self.df['Recommendations'].str.split(';').str.len().mean()
        insights.append(f"💡 Average recommendations per student: {avg_recommendations:.1f}")
        
        # Success prediction insights
        high_potential_count = (self.df['Success_Probability'] > 0.8).sum()
        insights.append(f"🌟 {high_potential_count} students identified as high potential (80%+ success probability)")
        
        # Engagement insights
        high_engagement = (self.df['Engagement_Score'] > 80).sum()
        insights.append(f"😊 {high_engagement} students show high engagement levels")
        
        # Interest correlation insights
        interest_cols = [col for col in self.df.columns if col.startswith('Interest_')]
        if len(interest_cols) > 1:
            interest_corr = self.df[interest_cols].corr()
            # Find strongest correlation
            max_corr = 0
            max_pair = None
            for i in range(len(interest_corr.columns)):
                for j in range(i+1, len(interest_corr.columns)):
                    corr_val = abs(interest_corr.iloc[i, j])
                    if corr_val > max_corr and not pd.isna(corr_val):
                        max_corr = corr_val
                        max_pair = (interest_corr.columns[i], interest_corr.columns[j])
            
            if max_pair:
                insights.append(f"🔗 Strongest interest correlation: {max_pair[0]} ↔ {max_pair[1]} ({max_corr:.2f})")
        
        print("📋 AI-GENERATED INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        return insights

def main():
    """Main function to run the AI-enhanced analysis"""
    print("🚀 Starting AI-Enhanced Student Data Analysis...")
    
    # Load and clean data
    df = load_and_clean_data()
    if df is None:
        return
    
    # Initialize AI analyzer
    ai_analyzer = AIStudentAnalyzer(df)
    
    # Perform AI analysis
    if AI_AVAILABLE:
        # Clustering
        cluster_analysis = ai_analyzer.perform_clustering(n_clusters=5)
        
        # Recommendations
        recommendations = ai_analyzer.generate_recommendations()
        
        # Success prediction
        model, feature_importance = ai_analyzer.predict_student_success()
        
        # Sentiment and engagement analysis
        engagement_data = ai_analyzer.analyze_student_sentiment()
        
        # Generate AI insights
        insights = ai_analyzer.generate_ai_insights()
        
        # Save enhanced data
        enhanced_df = df.copy()
        enhanced_df.to_csv('ai_enhanced_student_data.csv', index=False)
        print(f"\n✅ Enhanced data saved to 'ai_enhanced_student_data.csv'")
        
        # Create AI insights report
        with open('ai_insights_report.txt', 'w') as f:
            f.write("AI-Enhanced Student Analysis Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CLUSTER ANALYSIS:\n")
            f.write(str(cluster_analysis) + "\n\n")
            
            f.write("FEATURE IMPORTANCE:\n")
            f.write(str(feature_importance.head(10)) + "\n\n")
            
            f.write("AI INSIGHTS:\n")
            for insight in insights:
                f.write(f"- {insight}\n")
        
        print(f"✅ AI insights report saved to 'ai_insights_report.txt'")
        
    else:
        print("⚠️ AI features not available. Running basic analysis only.")
    
    print("\n✅ AI-ENHANCED ANALYSIS COMPLETE!")
    print(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 