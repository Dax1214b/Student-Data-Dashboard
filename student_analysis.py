#!/usr/bin/env python3
"""
Student Data Analysis Dashboard
A comprehensive Python script for analyzing student data with pandas, numpy, and matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("✅ Student Data Analysis Dashboard")
print("=" * 50)
print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_and_clean_data():
    """Load and clean the student data"""
    print("\n📁 Loading and cleaning data...")
    
    try:
        # Load the CSV file
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
    
    print("✅ Data cleaned successfully!")
    print(f"📧 Valid emails: {df['Valid_Email'].sum()} / {len(df)}")
    print(f"📅 Years standardized: {df['Year_Clean'].nunique()} unique values")
    print(f"🎯 Average interests per student: {df['Interest_Count'].mean():.1f}")
    
    return df

def generate_statistics(df):
    """Generate comprehensive statistics"""
    print("\n📊 GENERATING STATISTICS")
    print("=" * 50)
    
    # Basic counts
    total_students = len(df)
    total_branches = df['Branch'].nunique()
    total_years = df['Year_Clean'].nunique()
    valid_emails = df['Valid_Email'].sum()
    invalid_emails = (~df['Valid_Email']).sum()
    
    print(f"👥 Total Students: {total_students}")
    print(f"🏫 Total Branches: {total_branches}")
    print(f"📅 Total Years: {total_years}")
    print(f"📧 Valid Emails: {valid_emails}")
    print(f"❌ Invalid Emails: {invalid_emails}")
    
    # Branch distribution
    print("\n🏫 BRANCH DISTRIBUTION:")
    branch_counts = df['Branch'].value_counts()
    for branch, count in branch_counts.items():
        percentage = (count / total_students) * 100
        print(f"  {branch}: {count} students ({percentage:.1f}%)")
    
    # Year distribution
    print("\n📅 YEAR DISTRIBUTION:")
    year_counts = df['Year_Clean'].value_counts().sort_index()
    for year, count in year_counts.items():
        percentage = (count / total_students) * 100
        print(f"  Year {year}: {count} students ({percentage:.1f}%)")
    
    # Interest analysis
    print("\n🎯 TOP 10 INTERESTS:")
    all_interests = []
    for interests in df['Interests_List']:
        all_interests.extend(interests)
    
    interest_counts = pd.Series(all_interests).value_counts()
    for i, (interest, count) in enumerate(interest_counts.head(10).items(), 1):
        print(f"  {i:2d}. {interest}: {count} students")
    
    return {
        'total_students': total_students,
        'branch_counts': branch_counts,
        'year_counts': year_counts,
        'interest_counts': interest_counts
    }

def create_visualizations(df, stats):
    """Create comprehensive visualizations"""
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Student Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Branch Distribution
    branch_counts = stats['branch_counts']
    axes[0, 0].pie(branch_counts.values, labels=branch_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Branch Distribution')
    
    # 2. Year Distribution
    year_counts = stats['year_counts']
    axes[0, 1].bar(year_counts.index, year_counts.values, color='skyblue')
    axes[0, 1].set_title('Year Distribution')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of Students')
    
    # 3. Email Validity
    email_validity = df['Valid_Email'].value_counts()
    axes[0, 2].pie(email_validity.values, labels=['Invalid', 'Valid'], autopct='%1.1f%%', colors=['red', 'green'])
    axes[0, 2].set_title('Email Validity')
    
    # 4. Interest Count Distribution
    axes[1, 0].hist(df['Interest_Count'], bins=10, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Interest Count Distribution')
    axes[1, 0].set_xlabel('Number of Interests')
    axes[1, 0].set_ylabel('Number of Students')
    
    # 5. Branch vs Year Heatmap
    branch_year = pd.crosstab(df['Branch'], df['Year_Clean'])
    sns.heatmap(branch_year, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Branch vs Year Distribution')
    
    # 6. Top Interests
    interest_counts = stats['interest_counts'].head(10)
    axes[1, 2].barh(range(len(interest_counts)), interest_counts.values, color='orange')
    axes[1, 2].set_yticks(range(len(interest_counts)))
    axes[1, 2].set_yticklabels(interest_counts.index)
    axes[1, 2].set_title('Top 10 Interests')
    axes[1, 2].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('student_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'student_analysis_dashboard.png'")

class StudentDataAnalyzer:
    """Interactive class for filtering and analyzing student data"""
    
    def __init__(self, df):
        self.df = df
        self.filtered_df = df.copy()
    
    def search_students(self, query='', branch='', year='', interest=''):
        """Efficiently search and filter students based on multiple criteria"""
        mask = pd.Series([True] * len(self.df))

        # Text search across name, email, and interests
        if query:
            query_mask = (
                self.df['Name'].str.contains(query, case=False, na=False) |
                self.df['Email'].str.contains(query, case=False, na=False) |
                self.df['Interests'].str.contains(query, case=False, na=False)
            )
            mask &= query_mask

        # Branch filter
        if branch:
            mask &= (self.df['Branch'] == branch)

        # Year filter (handle 'second' and numeric)
        if year:
            if year == '2':
                year_mask = (
                    (self.df['Year_Clean'] == '2') |
                    (self.df['Year'].astype(str).str.contains('second', case=False, na=False))
                )
                mask &= year_mask
            else:
                mask &= (self.df['Year_Clean'] == year)

        # Interest filter (vectorized with str.contains on a joined string)
        if interest:
            interest_mask = self.df['Interests_List'].apply(
                lambda x: any(interest.lower() in str(i).lower() for i in x)
            )
            mask &= interest_mask

        self.filtered_df = self.df[mask]
        return self.filtered_df
    
    def get_filter_summary(self):
        """Get summary of current filtered data"""
        return {
            'Total_Students': len(self.filtered_df),
            'Branches': self.filtered_df['Branch'].unique().tolist(),
            'Years': self.filtered_df['Year_Clean'].unique().tolist(),
            'Valid_Emails': self.filtered_df['Valid_Email'].sum(),
            'Avg_Interests': self.filtered_df['Interest_Count'].mean()
        }
    
    def export_filtered_data(self, filename=None):
        """Export filtered data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'filtered_students_{timestamp}.csv'
        
        # Select relevant columns for export
        export_cols = ['Name', 'Branch', 'Year', 'Email', 'Interests']
        export_df = self.filtered_df[export_cols].copy()
        export_df.to_csv(filename, index=False)
        print(f"✅ Data exported to {filename}")
        return filename

class EmailManager:
    """Simulate email management functionality"""
    
    def __init__(self, df):
        self.df = df
        self.email_log = []
    
    def select_students_by_criteria(self, criteria):
        """Select students based on criteria"""
        analyzer = StudentDataAnalyzer(self.df)
        return analyzer.search_students(**criteria)
    
    def generate_email_template(self, template_type='general'):
        """Generate email templates"""
        templates = {
            'general': {
                'subject': 'Important Information for {name}',
                'body': '''Dear {name},

This is a general communication regarding your academic progress in {branch} (Year {year}).

Best regards,
Student Affairs Team'''
            },
            'ai_ml': {
                'subject': 'AI/ML Workshop Invitation for {name}',
                'body': '''Dear {name},

We noticed your interest in AI/ML and would like to invite you to our upcoming workshop.
As a {branch} student in year {year}, this would be perfect for you.

Best regards,
AI/ML Department'''
            },
            'web_dev': {
                'subject': 'Web Development Opportunity for {name}',
                'body': '''Dear {name},

Your interest in Web Development caught our attention!
We have exciting opportunities for {branch} students like you.

Best regards,
Web Development Team'''
            }
        }
        return templates.get(template_type, templates['general'])
    
    def send_personalized_emails(self, students, template_type='general'):
        """Simulate sending personalized emails"""
        template = self.generate_email_template(template_type)
        sent_count = 0
        
        for _, student in students.iterrows():
            if student['Valid_Email']:
                # Personalize the email
                subject = template['subject'].format(
                    name=student['Name'],
                    branch=student['Branch'],
                    year=student['Year_Clean']
                )
                
                body = template['body'].format(
                    name=student['Name'],
                    branch=student['Branch'],
                    year=student['Year_Clean']
                )
                
                # Log the email (simulation)
                self.email_log.append({
                    'to': student['Email'],
                    'subject': subject,
                    'template': template_type,
                    'timestamp': datetime.now()
                })
                
                sent_count += 1
        
        return sent_count, len(students)
    
    def get_email_statistics(self):
        """Get email sending statistics"""
        if not self.email_log:
            return {"Total_Emails": 0, "Templates_Used": [], "Recent_Emails": []}
        
        df_log = pd.DataFrame(self.email_log)
        return {
            "Total_Emails": len(df_log),
            "Templates_Used": df_log['template'].value_counts().to_dict(),
            "Recent_Emails": df_log.tail(5).to_dict('records')
        }

def advanced_analytics(df):
    """Perform advanced analytics and generate insights"""
    print("\n🔍 ADVANCED ANALYTICS AND INSIGHTS")
    print("=" * 50)
    
    # 1. Interest correlation analysis
    print("\n1. 🎯 INTEREST CORRELATION ANALYSIS:")
    
    # Create interest matrix
    all_interests = set()
    for interests in df['Interests_List']:
        all_interests.update(interests)
    
    interest_matrix = pd.DataFrame(index=df.index, columns=list(all_interests))
    for idx, interests in enumerate(df['Interests_List']):
        for interest in interests:
            interest_matrix.loc[idx, interest] = 1
    interest_matrix = interest_matrix.fillna(0)
    
    # Find correlations
    correlations = interest_matrix.corr()
    
    # Find top correlations
    top_correlations = []
    for i in range(len(correlations.columns)):
        for j in range(i+1, len(correlations.columns)):
            corr_value = correlations.iloc[i, j]
            if corr_value > 0.3:  # Threshold for meaningful correlation
                top_correlations.append({
                    'Interest1': correlations.columns[i],
                    'Interest2': correlations.columns[j],
                    'Correlation': corr_value
                })
    
    top_correlations.sort(key=lambda x: x['Correlation'], reverse=True)
    print("Top interest correlations:")
    for corr in top_correlations[:5]:
        print(f"  {corr['Interest1']} ↔ {corr['Interest2']}: {corr['Correlation']:.3f}")
    
    # 2. Branch analysis
    print("\n2. 🏫 BRANCH ANALYSIS:")
    branch_stats = df.groupby('Branch').agg({
        'Interest_Count': ['mean', 'std'],
        'Valid_Email': 'sum',
        'Student_ID': 'count'
    }).round(2)
    
    branch_stats.columns = ['Avg_Interests', 'Std_Interests', 'Valid_Emails', 'Total_Students']
    print(branch_stats)
    
    # 3. Year progression analysis
    print("\n3. 📅 YEAR PROGRESSION ANALYSIS:")
    year_stats = df.groupby('Year_Clean').agg({
        'Interest_Count': ['mean', 'count'],
        'Valid_Email': 'sum'
    }).round(2)
    
    year_stats.columns = ['Avg_Interests', 'Student_Count', 'Valid_Emails']
    print(year_stats)
    
    # 4. Data quality insights
    print("\n4. 📊 DATA QUALITY INSIGHTS:")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    print(f"Missing values per column:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # Email validity by branch
    email_by_branch = df.groupby('Branch')['Valid_Email'].agg(['sum', 'count'])
    email_by_branch['validity_rate'] = (email_by_branch['sum'] / email_by_branch['count'] * 100).round(1)
    print(f"\nEmail validity rate by branch:")
    print(email_by_branch[['validity_rate']])
    
    return {
        'interest_correlations': top_correlations,
        'branch_stats': branch_stats,
        'year_stats': year_stats,
        'email_quality': email_by_branch
    }

def interactive_demo(df):
    """Demonstrate interactive features"""
    print("\n🎛️ INTERACTIVE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer and email manager
    analyzer = StudentDataAnalyzer(df)
    email_manager = EmailManager(df)
    
    print("✅ Analyzer and Email Manager initialized!")
    
    # Example searches
    print("\n🔍 EXAMPLE SEARCHES:")
    
    # Search for CSE students
    cse_students = analyzer.search_students(branch='CSE')
    print(f"📊 CSE Students: {len(cse_students)} found")
    
    # Search for students with AI/ML interest
    ai_students = analyzer.search_students(interest='AI/ML')
    print(f"🎯 Students with AI/ML interest: {len(ai_students)} found")
    
    # Search for year 2 students (including 'second')
    year2_students = analyzer.search_students(year='2')
    print(f"📅 Year 2 students (including 'second'): {len(year2_students)} found")
    
    # Combined search
    combined = analyzer.search_students(branch='IT', year='2', interest='Web Dev')
    print(f"🔍 IT Year 2 students with Web Dev interest: {len(combined)} found")
    
    # Show summary
    print("\n📊 CURRENT FILTER SUMMARY:")
    summary = analyzer.get_filter_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Email examples
    print("\n📧 EMAIL EXAMPLES:")
    
    # Send emails to AI/ML students
    ai_students = email_manager.select_students_by_criteria({'interest': 'AI/ML'})
    sent, total = email_manager.send_personalized_emails(ai_students, 'ai_ml')
    print(f"📧 Sent {sent} emails to {total} AI/ML students")
    
    # Send emails to Web Dev students
    web_students = email_manager.select_students_by_criteria({'interest': 'Web Dev'})
    sent2, total2 = email_manager.send_personalized_emails(web_students, 'web_dev')
    print(f"📧 Sent {sent2} emails to {total2} Web Dev students")
    
    # Show email statistics
    print("\n📊 EMAIL STATISTICS:")
    email_stats = email_manager.get_email_statistics()
    for key, value in email_stats.items():
        print(f"{key}: {value}")
    
    # Export example
    print("\n📥 EXPORT EXAMPLE:")
    analyzer.search_students(branch='CSE', year='2')
    export_file = analyzer.export_filtered_data()
    print(f"Exported CSE Year 2 students to: {export_file}")
    
    return analyzer, email_manager

def main():
    """Main function to run the complete analysis"""
    print("🚀 Starting Student Data Analysis...")
    
    # Load and clean data
    df = load_and_clean_data()
    if df is None:
        return
    
    # Generate statistics
    stats = generate_statistics(df)
    
    # Create visualizations
    create_visualizations(df, stats)
    
    # Advanced analytics
    insights = advanced_analytics(df)
    
    # Interactive demo
    analyzer, email_manager = interactive_demo(df)
    
    print("\n✅ ANALYSIS COMPLETE!")
    print(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 You can now use the analyzer and email_manager objects for further analysis!")
    print("\n📋 USAGE EXAMPLES:")
    print("• analyzer.search_students(branch='CSE')")
    print("• analyzer.search_students(year='2')")
    print("• analyzer.search_students(interest='AI/ML')")
    print("• analyzer.export_filtered_data('my_filtered_data.csv')")
    print("• email_manager.send_personalized_emails(students, 'ai_ml')")

if __name__ == "__main__":
    main()