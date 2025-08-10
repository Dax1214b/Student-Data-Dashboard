#!/usr/bin/env python3
"""
Student Data Analysis Chatbot
Interactive chatbot for querying and analyzing student data using natural language.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StudentDataChatbot:
    """Interactive chatbot for student data analysis"""
    
    def __init__(self):
        self.df = None
        self.load_data()
        self.commands = {
            'help': self.show_help,
            'stats': self.show_statistics,
            'search': self.search_students,
            'filter': self.filter_students,
            'cluster': self.show_clusters,
            'recommend': self.get_recommendations,
            'export': self.export_data,
            'quit': self.quit_chatbot
        }
    
    def load_data(self):
        """Load and clean student data"""
        try:
            self.df = pd.read_csv('student_data_with_edge_cases.xlsx - Sheet1.csv')
            self.df = self.df.fillna('')
            
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
            
            self.df['Year_Clean'] = self.df['Year'].apply(standardize_year)
            
            # Clean Email column
            def is_valid_email(email):
                if pd.isna(email) or email == '':
                    return False
                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                return bool(re.match(pattern, str(email)))
            
            self.df['Valid_Email'] = self.df['Email'].apply(is_valid_email)
            
            # Clean Interests
            def clean_interests(interests):
                if pd.isna(interests) or interests == '':
                    return []
                interest_list = [interest.strip() for interest in str(interests).split(',')]
                return [interest for interest in interest_list if interest]
            
            self.df['Interests_List'] = self.df['Interests'].apply(clean_interests)
            self.df['Interest_Count'] = self.df['Interests_List'].apply(len)
            self.df['Student_ID'] = range(1, len(self.df) + 1)
            
            print("✅ Student data loaded successfully!")
            
        except FileNotFoundError:
            print("❌ Student data file not found!")
            self.df = None
    
    def show_help(self, query=""):
        """Show available commands"""
        print("\n🤖 STUDENT DATA CHATBOT - Available Commands:")
        print("=" * 50)
        print("📊 stats                    - Show general statistics")
        print("🔍 search [name/email]      - Search for specific students")
        print("🎯 filter [branch/year/interest] - Filter students by criteria")
        print("📈 cluster                  - Show student clusters")
        print("💡 recommend [student]      - Get recommendations for student")
        print("📥 export [filename]        - Export filtered data")
        print("❓ help                     - Show this help message")
        print("🚪 quit                     - Exit chatbot")
        print("\n💬 You can also ask natural language questions like:")
        print("   • 'How many CSE students are there?'")
        print("   • 'Show me students interested in AI/ML'")
        print("   • 'What are the most popular interests?'")
        print("   • 'Find students in year 1/2/3/4/5'")
        print("   • 'Show me first year students'")
        print("   • 'Find third year students'")
    
    def show_statistics(self, query=""):
        """Show comprehensive statistics"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        print("\n📊 STUDENT DATA STATISTICS")
        print("=" * 40)
        
        # Basic stats
        total_students = len(self.df)
        valid_emails = self.df['Valid_Email'].sum()
        avg_interests = self.df['Interest_Count'].mean()
        
        print(f"👥 Total Students: {total_students}")
        print(f"📧 Valid Emails: {valid_emails} ({valid_emails/total_students*100:.1f}%)")
        print(f"🎯 Average Interests: {avg_interests:.1f} per student")
        
        # Branch distribution
        print(f"\n🏫 BRANCH DISTRIBUTION:")
        branch_counts = self.df['Branch'].value_counts()
        for branch, count in branch_counts.items():
            percentage = (count / total_students) * 100
            print(f"  {branch}: {count} students ({percentage:.1f}%)")
        
        # Year distribution
        print(f"\n📅 YEAR DISTRIBUTION:")
        year_counts = self.df['Year_Clean'].value_counts().sort_index()
        for year, count in year_counts.items():
            percentage = (count / total_students) * 100
            print(f"  Year {year}: {count} students ({percentage:.1f}%)")
        
        # Top interests
        print(f"\n🎯 TOP 10 INTERESTS:")
        all_interests = []
        for interests in self.df['Interests_List']:
            all_interests.extend(interests)
        
        interest_counts = pd.Series(all_interests).value_counts()
        for i, (interest, count) in enumerate(interest_counts.head(10).items(), 1):
            print(f"  {i:2d}. {interest}: {count} students")
    
    def search_students(self, query=""):
        """Search for students by name or email"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        if not query:
            query = input("🔍 Enter name or email to search: ").strip()
        
        if not query:
            print("❌ Please provide a search term!")
            return
        
        # Search in name and email
        mask = (
            self.df['Name'].str.contains(query, case=False, na=False) |
            self.df['Email'].str.contains(query, case=False, na=False)
        )
        
        results = self.df[mask]
        
        if len(results) == 0:
            print(f"❌ No students found matching '{query}'")
            return
        
        print(f"\n🔍 Found {len(results)} student(s) matching '{query}':")
        print("=" * 60)
        
        for _, student in results.iterrows():
            print(f"\n👤 {student['Name']}")
            print(f"   🏫 Branch: {student['Branch']}")
            print(f"   📅 Year: {student['Year_Clean']}")
            print(f"   📧 Email: {student['Email']}")
            print(f"   🎯 Interests: {', '.join(student['Interests_List'])}")
            print(f"   ✅ Valid Email: {'Yes' if student['Valid_Email'] else 'No'}")
    
    def filter_students(self, query=""):
        """Filter students by various criteria"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        if not query:
            print("\n🎯 FILTER OPTIONS:")
            print("1. Branch (e.g., 'CSE', 'IT', 'MECH')")
            print("2. Year (e.g., '1', '2', '3', '4', '5')")
            print("3. Interest (e.g., 'AI/ML', 'Web Dev', 'Cybersecurity')")
            query = input("Enter filter criteria: ").strip()
        
        if not query:
            print("❌ Please provide filter criteria!")
            return
        
        filtered_df = self.df.copy()
        
        # Branch filter
        if query.upper() in ['CSE', 'IT', 'MECH', 'ECE', 'EEE', 'CIVIL', 'MBA', 'LAW']:
            filtered_df = filtered_df[filtered_df['Branch'] == query.upper()]
            filter_type = "Branch"
        
        # Year filter
        elif query in ['1', '2', '3', '4', '5']:
            if query == '2':
                # Include both '2' and 'second'
                mask = (
                    (filtered_df['Year_Clean'] == '2') |
                    (filtered_df['Year'].astype(str).str.contains('second', case=False, na=False))
                )
                filtered_df = filtered_df[mask]
            else:
                # For years 1, 3, 4, 5 - handle both numeric and text versions
                mask = (
                    (filtered_df['Year_Clean'] == query) |
                    (filtered_df['Year'].astype(str) == query)
                )
                filtered_df = filtered_df[mask]
            filter_type = "Year"
        
        # Interest filter
        else:
            mask = filtered_df['Interests_List'].apply(
                lambda x: any(query.lower() in str(i).lower() for i in x)
            )
            filtered_df = filtered_df[mask]
            filter_type = "Interest"
        
        if len(filtered_df) == 0:
            print(f"❌ No students found with {filter_type.lower()} '{query}'")
            return
        
        print(f"\n🎯 Found {len(filtered_df)} student(s) with {filter_type.lower()} '{query}':")
        print("=" * 60)
        
        # Show summary
        print(f"📊 Summary:")
        print(f"  • Branches: {', '.join(filtered_df['Branch'].unique())}")
        print(f"  • Years: {', '.join(filtered_df['Year_Clean'].unique())}")
        print(f"  • Valid Emails: {filtered_df['Valid_Email'].sum()}/{len(filtered_df)}")
        print(f"  • Avg Interests: {filtered_df['Interest_Count'].mean():.1f}")
        
        # Show first 5 students
        print(f"\n👥 First 5 students:")
        for _, student in filtered_df.head(5).iterrows():
            print(f"  • {student['Name']} ({student['Branch']}, Year {student['Year_Clean']})")
        
        if len(filtered_df) > 5:
            print(f"  ... and {len(filtered_df) - 5} more students")
        
        # Store filtered data for export
        self.current_filtered = filtered_df
    
    def show_clusters(self, query=""):
        """Show student clusters (simplified version)"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        print("\n📈 STUDENT CLUSTERS (Based on Interest Count)")
        print("=" * 50)
        
        # Simple clustering based on interest count
        self.df['Cluster'] = pd.cut(
            self.df['Interest_Count'],
            bins=[0, 1, 2, 3, 10],
            labels=['Low Interest', 'Medium Interest', 'High Interest', 'Very High Interest']
        )
        
        cluster_analysis = self.df.groupby('Cluster').agg({
            'Interest_Count': ['mean', 'count'],
            'Branch': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        cluster_analysis.columns = ['Avg_Interests', 'Student_Count', 'Most_Common_Branch']
        
        for cluster_name, data in cluster_analysis.iterrows():
            print(f"\n🎯 {cluster_name}:")
            print(f"  👥 Students: {data['Student_Count']}")
            print(f"  🎯 Avg Interests: {data['Avg_Interests']}")
            print(f"  🏫 Common Branch: {data['Most_Common_Branch']}")
    
    def get_recommendations(self, query=""):
        """Get recommendations for students"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        if not query:
            query = input("👤 Enter student name for recommendations: ").strip()
        
        if not query:
            print("❌ Please provide a student name!")
            return
        
        # Find student
        student = self.df[self.df['Name'].str.contains(query, case=False, na=False)]
        
        if len(student) == 0:
            print(f"❌ Student '{query}' not found!")
            return
        
        student = student.iloc[0]
        
        print(f"\n💡 RECOMMENDATIONS FOR {student['Name']}")
        print("=" * 50)
        print(f"🏫 Branch: {student['Branch']}")
        print(f"📅 Year: {student['Year_Clean']}")
        print(f"🎯 Interests: {', '.join(student['Interests_List'])}")
        
        # Generate recommendations based on interests
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
        
        print(f"\n💡 RECOMMENDED ACTIVITIES:")
        student_recommendations = []
        
        for interest in student['Interests_List']:
            if interest in recommendations:
                student_recommendations.extend(recommendations[interest])
        
        # Remove duplicates and show top 5
        student_recommendations = list(set(student_recommendations))[:5]
        
        for i, rec in enumerate(student_recommendations, 1):
            print(f"  {i}. {rec}")
        
        if not student_recommendations:
            print("  No specific recommendations available for current interests.")
    
    def export_data(self, query=""):
        """Export filtered data to CSV"""
        if self.df is None:
            print("❌ No data loaded!")
            return
        
        if not query:
            query = input("📁 Enter filename for export (or press Enter for default): ").strip()
        
        if not query:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            query = f'student_export_{timestamp}.csv'
        
        if not query.endswith('.csv'):
            query += '.csv'
        
        # Export current filtered data or all data
        export_df = getattr(self, 'current_filtered', self.df)
        
        # Select relevant columns
        export_cols = ['Name', 'Branch', 'Year', 'Email', 'Interests']
        export_df = export_df[export_cols].copy()
        
        export_df.to_csv(query, index=False)
        print(f"✅ Data exported to '{query}' ({len(export_df)} students)")
    
    def quit_chatbot(self, query=""):
        """Exit the chatbot"""
        print("\n👋 Thank you for using the Student Data Chatbot!")
        print("📊 Analysis completed at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return True
    
    def process_natural_language(self, query):
        """Process natural language queries"""
        query_lower = query.lower()
        
        # Statistics queries
        if any(word in query_lower for word in ['how many', 'total', 'statistics', 'stats']):
            if 'cse' in query_lower:
                cse_count = len(self.df[self.df['Branch'] == 'CSE'])
                print(f"📊 There are {cse_count} CSE students.")
            elif 'year' in query_lower and any(str(i) in query_lower for i in range(1, 6)):
                # Show year-specific stats
                for year in range(1, 6):
                    if str(year) in query_lower:
                        if year == 2:
                            # Include both '2' and 'second'
                            mask = (
                                (self.df['Year_Clean'] == '2') |
                                (self.df['Year'].astype(str).str.contains('second', case=False, na=False))
                            )
                            year_count = len(self.df[mask])
                        else:
                            year_count = len(self.df[self.df['Year_Clean'] == str(year)])
                        print(f"📊 There are {year_count} students in Year {year}.")
                        break
            elif 'student' in query_lower:
                print(f"📊 There are {len(self.df)} total students.")
            else:
                self.show_statistics()
            return True
        
        # Search queries
        if any(word in query_lower for word in ['find', 'search', 'show me', 'who']):
            if 'ai/ml' in query_lower or 'ai' in query_lower:
                self.filter_students('AI/ML')
            elif 'web dev' in query_lower:
                self.filter_students('Web Dev')
            elif 'year 1' in query_lower or 'first year' in query_lower:
                self.filter_students('1')
            elif 'year 2' in query_lower or 'second year' in query_lower:
                self.filter_students('2')
            elif 'year 3' in query_lower or 'third year' in query_lower:
                self.filter_students('3')
            elif 'year 4' in query_lower or 'fourth year' in query_lower:
                self.filter_students('4')
            elif 'year 5' in query_lower or 'fifth year' in query_lower:
                self.filter_students('5')
            else:
                self.search_students()
            return True
        
        # Interest queries
        if 'interest' in query_lower or 'popular' in query_lower:
            all_interests = []
            for interests in self.df['Interests_List']:
                all_interests.extend(interests)
            
            interest_counts = pd.Series(all_interests).value_counts()
            print(f"\n🎯 MOST POPULAR INTERESTS:")
            for i, (interest, count) in enumerate(interest_counts.head(5).items(), 1):
                print(f"  {i}. {interest}: {count} students")
            return True
        
        return False
    
    def run(self):
        """Main chatbot loop"""
        print("🤖 Welcome to the Student Data Analysis Chatbot!")
        print("=" * 60)
        print("💬 Ask me anything about the student data!")
        print("📋 Type 'help' to see available commands")
        print("🚪 Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n🤖 You: ").strip()
                
                if not query:
                    continue
                
                # Check for direct commands
                command = query.split()[0].lower()
                if command in self.commands:
                    if self.commands[command](query[len(command):].strip()):
                        break
                    continue
                
                # Try natural language processing
                if self.process_natural_language(query):
                    continue
                
                # Default response
                print("❓ I didn't understand that. Type 'help' to see available commands.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    chatbot = StudentDataChatbot()
    chatbot.run()

if __name__ == "__main__":
    main() 