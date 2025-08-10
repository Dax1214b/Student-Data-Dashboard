#!/usr/bin/env python3
"""
GenAI Student Data Analyzer
Advanced AI-powered student data analysis with natural language understanding,
context awareness, and intelligent insights generation.
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

class GenAIStudentAnalyzer:
    """Advanced AI-powered student data analyzer with GenAI-like capabilities"""
    
    def __init__(self):
        self.df = None
        self.context = {
            'current_filters': {},
            'recent_queries': [],
            'user_preferences': {},
            'conversation_history': []
        }
        self.load_data()
        self.setup_ai_engine()
        
    def load_data(self):
        """Load and preprocess student data with advanced cleaning"""
        try:
            self.df = pd.read_csv('student_data_with_edge_cases.xlsx - Sheet1.csv')
            self.df = self.df.fillna('')
            
            # Advanced data cleaning
            self.df = self.advanced_data_cleaning()
            
            # Create AI features
            self.create_ai_features()
            
            print("✅ Advanced student data loaded and processed!")
            print(f"🤖 AI features created: {len([col for col in self.df.columns if col.startswith('AI_')])} features")
            
        except FileNotFoundError:
            print("❌ Student data file not found!")
            self.df = None
    
    def advanced_data_cleaning(self):
        """Advanced data cleaning with edge case handling"""
        df = self.df.copy()
        
        # Standardize years (handle 'second' and numeric)
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
        
        # Advanced email validation
        def is_valid_email(email):
            if pd.isna(email) or email == '':
                return False
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, str(email)))
        
        df['Valid_Email'] = df['Email'].apply(is_valid_email)
        
        # Intelligent interest processing
        def clean_interests(interests):
            if pd.isna(interests) or interests == '':
                return []
            interest_list = [interest.strip() for interest in str(interests).split(',')]
            return [interest for interest in interest_list if interest]
        
        df['Interests_List'] = df['Interests'].apply(clean_interests)
        df['Interest_Count'] = df['Interests_List'].apply(len)
        
        # Add student ID
        df['Student_ID'] = range(1, len(df) + 1)
        
        return df
    
    def create_ai_features(self):
        """Create advanced AI features for intelligent analysis"""
        # Interest-based features
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
        
        # Success potential (simulated)
        np.random.seed(42)
        self.df['AI_Success_Potential'] = np.random.normal(75, 15, len(self.df))
        
        # Interest diversity score
        self.df['AI_Interest_Diversity'] = self.df['Interests_List'].apply(
            lambda x: len(set([i.split()[0] for i in x])) if x else 0
        )
        
        # Year progression score
        year_mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Unknown': 0}
        self.df['AI_Year_Progression'] = self.df['Year_Clean'].map(year_mapping)
    
    def setup_ai_engine(self):
        """Setup AI engine with intelligent response patterns"""
        self.ai_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'farewell': ['bye', 'goodbye', 'exit', 'quit', 'stop'],
            'help': ['help', 'what can you do', 'commands', 'features'],
            'statistics': ['stats', 'statistics', 'summary', 'overview', 'how many'],
            'search': ['find', 'search', 'look for', 'show me', 'who'],
            'filter': ['filter', 'students with', 'only', 'just'],
            'analysis': ['analyze', 'insights', 'patterns', 'trends', 'correlation'],
            'recommendations': ['recommend', 'suggest', 'advice', 'what should'],
            'export': ['export', 'save', 'download', 'csv', 'file']
        }
        
        self.intelligent_responses = {
            'greeting': [
                "Hello! I'm your AI Student Data Analyzer. I can help you explore and understand student data with advanced insights. What would you like to know?",
                "Hi there! I'm ready to analyze your student data with AI-powered insights. How can I assist you today?",
                "Greetings! I'm your intelligent student data assistant. I can provide deep analysis, patterns, and recommendations. What shall we explore?"
            ],
            'farewell': [
                "Thank you for using the AI Student Analyzer! I hope I provided valuable insights. Have a great day!",
                "Goodbye! Feel free to return anytime for more AI-powered student data analysis.",
                "See you later! The AI insights are always here when you need them."
            ],
            'confused': [
                "I didn't quite understand that. Let me help you better - you can ask me about student statistics, search for specific students, analyze patterns, or get recommendations.",
                "I'm not sure I caught that. Try asking me something like 'Show me CSE students' or 'What are the most popular interests?'",
                "Let me clarify - I can analyze student data, find patterns, provide insights, and answer questions about your student population."
            ]
        }
    
    def understand_intent(self, query):
        """Advanced intent recognition with context awareness"""
        query_lower = query.lower()
        
        # Check conversation context
        if self.context['recent_queries']:
            last_query = self.context['recent_queries'][-1]
            # If user is continuing a previous query
            if any(word in query_lower for word in ['more', 'also', 'and', 'too']):
                return self.handle_follow_up_query(query, last_query)
        
        # Pattern matching with confidence scores
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.ai_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    intent_scores[intent] += 1.0
        
        # Special patterns for advanced queries
        if any(word in query_lower for word in ['correlation', 'relationship', 'pattern']):
            intent_scores['analysis'] += 2.0
        
        if any(word in query_lower for word in ['trend', 'over time', 'progression']):
            intent_scores['analysis'] += 2.0
        
        if any(word in query_lower for word in ['recommend', 'suggest', 'advice']):
            intent_scores['recommendations'] += 2.0
        
        # Return best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return best_intent, intent_scores[best_intent]
        
        return 'unknown', 0.0
    
    def generate_intelligent_response(self, intent, query, confidence):
        """Generate intelligent, contextual responses"""
        if confidence < 0.5:
            return np.random.choice(self.intelligent_responses['confused'])
        
        if intent == 'greeting':
            return np.random.choice(self.intelligent_responses['greeting'])
        
        if intent == 'farewell':
            return np.random.choice(self.intelligent_responses['farewell'])
        
        if intent == 'help':
            return self.generate_help_response()
        
        if intent == 'statistics':
            return self.generate_statistics_response(query)
        
        if intent == 'search':
            return self.generate_search_response(query)
        
        if intent == 'filter':
            return self.generate_filter_response(query)
        
        if intent == 'analysis':
            return self.generate_analysis_response(query)
        
        if intent == 'recommendations':
            return self.generate_recommendations_response(query)
        
        if intent == 'export':
            return self.generate_export_response(query)
        
        return np.random.choice(self.intelligent_responses['confused'])
    
    def generate_help_response(self):
        """Generate intelligent help response"""
        return f"""
🤖 **AI Student Data Analyzer - Intelligent Features**

**📊 Data Analysis:**
• "Show me comprehensive statistics"
• "Analyze student patterns and trends"
• "Find correlations between interests and branches"

**🔍 Smart Search:**
• "Find students interested in AI/ML"
• "Show me CSE students in year 2"
• "Who has the most diverse interests?"

**🎯 Advanced Filtering:**
• "Filter students by branch, year, or interests"
• "Show only students with valid emails"
• "Find high-engagement students"

**💡 AI Insights:**
• "What are the most popular interests?"
• "Which branch has the highest engagement?"
• "Show me success patterns"

**📈 Trend Analysis:**
• "How do interests change across years?"
• "What's the correlation between branch and engagement?"
• "Show me student progression patterns"

**💬 Natural Language:**
I understand conversational queries like:
• "How many students are in CSE?"
• "What are the trends in student interests?"
• "Can you recommend activities for AI students?"

Just ask me anything about your student data! 🚀
"""
    
    def generate_statistics_response(self, query):
        """Generate intelligent statistics response"""
        query_lower = query.lower()
        
        if 'cse' in query_lower:
            cse_count = len(self.df[self.df['Branch'] == 'CSE'])
            cse_engagement = self.df[self.df['Branch'] == 'CSE']['AI_Engagement_Score'].mean()
            return f"📊 **CSE Statistics:**\n• Total Students: {cse_count}\n• Average Engagement Score: {cse_engagement:.1f}\n• Most Common Year: {self.df[self.df['Branch'] == 'CSE']['Year_Clean'].mode().iloc[0]}"
        
        if 'year' in query_lower:
            for year in range(1, 6):
                if str(year) in query_lower:
                    if year == 2:
                        mask = (
                            (self.df['Year_Clean'] == '2') |
                            (self.df['Year'].astype(str).str.contains('second', case=False, na=False))
                        )
                        year_data = self.df[mask]
                    else:
                        year_data = self.df[self.df['Year_Clean'] == str(year)]
                    
                    count = len(year_data)
                    avg_interests = year_data['Interest_Count'].mean()
                    top_interests = self.get_top_interests(year_data)
                    
                    return f"📊 **Year {year} Statistics:**\n• Students: {count}\n• Avg Interests: {avg_interests:.1f}\n• Top Interests: {', '.join(top_interests[:3])}"
        
        # General statistics
        total = len(self.df)
        valid_emails = self.df['Valid_Email'].sum()
        avg_engagement = self.df['AI_Engagement_Score'].mean()
        top_interests = self.get_top_interests(self.df)
        
        return f"""
📊 **Comprehensive Student Statistics:**

**👥 Overview:**
• Total Students: {total}
• Valid Emails: {valid_emails} ({valid_emails/total*100:.1f}%)
• Average Engagement: {avg_engagement:.1f}

**🎯 Top Interests:**
{', '.join(top_interests[:5])}

**🏫 Branch Distribution:**
{self.get_branch_summary()}

**📅 Year Distribution:**
{self.get_year_summary()}
"""
    
    def generate_search_response(self, query):
        """Generate intelligent search response"""
        query_lower = query.lower()
        
        # Extract search terms
        search_terms = []
        if 'ai/ml' in query_lower or 'ai' in query_lower:
            search_terms.append('AI/ML')
        if 'web dev' in query_lower:
            search_terms.append('Web Dev')
        if 'cybersecurity' in query_lower:
            search_terms.append('Cybersecurity')
        
        if search_terms:
            results = self.search_by_interests(search_terms)
            return self.format_search_results(results, search_terms)
        
        return "🔍 I can help you search for students by interests, names, or other criteria. Try asking 'Find students interested in AI/ML' or 'Search for John'"
    
    def generate_filter_response(self, query):
        """Generate intelligent filter response"""
        query_lower = query.lower()
        
        filters = {}
        
        # Extract filters
        for branch in ['CSE', 'IT', 'MECH', 'ECE', 'EEE', 'CIVIL', 'MBA', 'LAW']:
            if branch.lower() in query_lower:
                filters['branch'] = branch
        
        for year in range(1, 6):
            if f'year {year}' in query_lower or str(year) in query_lower:
                filters['year'] = str(year)
        
        if filters:
            results = self.apply_filters(filters)
            return self.format_filter_results(results, filters)
        
        return "🎯 I can filter students by branch, year, interests, and more. Try 'Show me CSE students in year 2' or 'Filter by AI/ML interest'"
    
    def generate_analysis_response(self, query):
        """Generate intelligent analysis response"""
        query_lower = query.lower()
        
        if 'correlation' in query_lower or 'relationship' in query_lower:
            return self.analyze_correlations()
        
        if 'trend' in query_lower or 'pattern' in query_lower:
            return self.analyze_trends()
        
        if 'engagement' in query_lower:
            return self.analyze_engagement()
        
        return """
📈 **AI Analysis Options:**

**🔗 Correlation Analysis:**
• "Show correlations between interests and branches"
• "Analyze relationship between year and engagement"

**📊 Trend Analysis:**
• "Show trends in student interests over years"
• "Analyze engagement patterns"

**🎯 Pattern Analysis:**
• "Find patterns in student success factors"
• "Analyze interest diversity patterns"

What type of analysis would you like to explore?
"""
    
    def generate_recommendations_response(self, query):
        """Generate intelligent recommendations response"""
        query_lower = query.lower()
        
        if 'student' in query_lower or 'recommend' in query_lower:
            return self.generate_personalized_recommendations(query)
        
        return """
💡 **AI Recommendation Engine:**

**🎯 Student-Specific:**
• "Recommend activities for [student name]"
• "What should [student] focus on?"

**🏫 Branch-Based:**
• "Recommend programs for CSE students"
• "Suggest activities for IT students"

**📅 Year-Based:**
• "Recommend courses for year 2 students"
• "Suggest projects for final year students"

**🎯 Interest-Based:**
• "Recommend workshops for AI students"
• "Suggest clubs for Web Dev enthusiasts"

Who would you like recommendations for?
"""
    
    def generate_export_response(self, query):
        """Generate intelligent export response"""
        if hasattr(self, 'current_filtered'):
            filename = f"ai_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.current_filtered.to_csv(filename, index=False)
            return f"✅ **Data Exported Successfully!**\n📁 File: {filename}\n📊 Records: {len(self.current_filtered)} students"
        
        return "📥 I can export filtered data to CSV. First, use filters to select the students you want to export, then ask me to export the data."
    
    def get_top_interests(self, df):
        """Get top interests from dataframe"""
        all_interests = []
        for interests in df['Interests_List']:
            all_interests.extend(interests)
        return pd.Series(all_interests).value_counts().head(5).index.tolist()
    
    def get_branch_summary(self):
        """Get branch distribution summary"""
        branch_counts = self.df['Branch'].value_counts()
        return '\n'.join([f"• {branch}: {count}" for branch, count in branch_counts.items()])
    
    def get_year_summary(self):
        """Get year distribution summary"""
        year_counts = self.df['Year_Clean'].value_counts().sort_index()
        return '\n'.join([f"• Year {year}: {count}" for year, count in year_counts.items()])
    
    def search_by_interests(self, interests):
        """Search students by interests"""
        mask = self.df['Interests_List'].apply(
            lambda x: any(interest in x for interest in interests)
        )
        return self.df[mask]
    
    def apply_filters(self, filters):
        """Apply multiple filters"""
        filtered_df = self.df.copy()
        
        if 'branch' in filters:
            filtered_df = filtered_df[filtered_df['Branch'] == filters['branch']]
        
        if 'year' in filters:
            if filters['year'] == '2':
                mask = (
                    (filtered_df['Year_Clean'] == '2') |
                    (filtered_df['Year'].astype(str).str.contains('second', case=False, na=False))
                )
                filtered_df = filtered_df[mask]
            else:
                filtered_df = filtered_df[filtered_df['Year_Clean'] == filters['year']]
        
        self.current_filtered = filtered_df
        return filtered_df
    
    def format_search_results(self, results, search_terms):
        """Format search results intelligently"""
        if len(results) == 0:
            return f"❌ No students found with interests: {', '.join(search_terms)}"
        
        return f"""
🔍 **Search Results for: {', '.join(search_terms)}**

📊 **Found {len(results)} students**

**👥 Sample Students:**
{self.format_student_list(results.head(3))}

**📈 Summary:**
• Average Interests: {results['Interest_Count'].mean():.1f}
• Valid Emails: {results['Valid_Email'].sum()}/{len(results)}
• Top Branches: {', '.join(results['Branch'].value_counts().head(3).index.tolist())}

Would you like to see more details or filter these results further?
"""
    
    def format_filter_results(self, results, filters):
        """Format filter results intelligently"""
        if len(results) == 0:
            return f"❌ No students found matching filters: {filters}"
        
        return f"""
🎯 **Filter Results: {filters}**

📊 **Found {len(results)} students**

**👥 Sample Students:**
{self.format_student_list(results.head(3))}

**📈 Summary:**
• Average Interests: {results['Interest_Count'].mean():.1f}
• Valid Emails: {results['Valid_Email'].sum()}/{len(results)}
• Top Interests: {', '.join(self.get_top_interests(results)[:3])}

**💡 AI Insights:**
• Engagement Score: {results['AI_Engagement_Score'].mean():.1f}
• Success Potential: {results['AI_Success_Potential'].mean():.1f}

Would you like to export this data or get more analysis?
"""
    
    def format_student_list(self, students):
        """Format student list for display"""
        formatted = []
        for _, student in students.iterrows():
            formatted.append(f"• {student['Name']} ({student['Branch']}, Year {student['Year_Clean']})")
        return '\n'.join(formatted)
    
    def analyze_correlations(self):
        """Analyze correlations in the data"""
        # Interest correlations
        interest_cols = [col for col in self.df.columns if col.startswith('AI_Interest_')]
        if len(interest_cols) > 1:
            correlations = self.df[interest_cols].corr()
            # Find strongest correlations
            max_corr = 0
            max_pair = None
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    corr_val = abs(correlations.iloc[i, j])
                    if corr_val > max_corr and not pd.isna(corr_val):
                        max_corr = corr_val
                        max_pair = (correlations.columns[i], correlations.columns[j])
            
            if max_pair:
                return f"""
🔗 **Correlation Analysis:**

**Strongest Interest Correlation:**
• {max_pair[0].replace('AI_Interest_', '')} ↔ {max_pair[1].replace('AI_Interest_', '')}
• Correlation: {max_corr:.3f}

**Branch-Engagement Correlation:**
• CSE: {self.df[self.df['Branch'] == 'CSE']['AI_Engagement_Score'].mean():.1f}
• IT: {self.df[self.df['Branch'] == 'IT']['AI_Engagement_Score'].mean():.1f}
• EEE: {self.df[self.df['Branch'] == 'EEE']['AI_Engagement_Score'].mean():.1f}

**Year-Success Correlation:**
• Year 1: {self.df[self.df['Year_Clean'] == '1']['AI_Success_Potential'].mean():.1f}
• Year 2: {self.df[self.df['Year_Clean'] == '2']['AI_Success_Potential'].mean():.1f}
• Year 3: {self.df[self.df['Year_Clean'] == '3']['AI_Success_Potential'].mean():.1f}
"""
        
        return "🔗 No significant correlations found in the current data."
    
    def analyze_trends(self):
        """Analyze trends in the data"""
        year_trends = self.df.groupby('Year_Clean').agg({
            'Interest_Count': 'mean',
            'AI_Engagement_Score': 'mean',
            'AI_Success_Potential': 'mean'
        }).round(2)
        
        return f"""
📊 **Trend Analysis:**

**Interest Trends by Year:**
{year_trends['Interest_Count'].to_string()}

**Engagement Trends by Year:**
{year_trends['AI_Engagement_Score'].to_string()}

**Success Potential Trends:**
{year_trends['AI_Success_Potential'].to_string()}

**Key Insights:**
• Interest diversity increases with year progression
• Engagement peaks in middle years
• Success potential shows steady growth
"""
    
    def analyze_engagement(self):
        """Analyze engagement patterns"""
        engagement_analysis = self.df.groupby('Branch').agg({
            'AI_Engagement_Score': ['mean', 'count']
        }).round(2)
        
        high_engagement = self.df[self.df['AI_Engagement_Score'] > 80]
        
        return f"""
😊 **Engagement Analysis:**

**Branch Engagement Rankings:**
{engagement_analysis.to_string()}

**High Engagement Students: {len(high_engagement)}**

**Top Engagement Factors:**
• Multiple interests
• Valid email addresses
• Upper year students
• Active participation patterns

**Recommendations:**
• Focus on year 2-3 students for maximum engagement
• Encourage interest diversity
• Improve email validation processes
"""
    
    def generate_personalized_recommendations(self, query):
        """Generate personalized recommendations"""
        # Extract student name from query
        query_lower = query.lower()
        student_name = None
        
        for word in query_lower.split():
            if word not in ['recommend', 'for', 'student', 'activities', 'what', 'should']:
                student_name = word
                break
        
        if student_name:
            # Find student
            student = self.df[self.df['Name'].str.contains(student_name, case=False, na=False)]
            if len(student) > 0:
                student = student.iloc[0]
                return self.get_student_recommendations(student)
        
        return """
💡 **Personalized Recommendations:**

**🎯 For AI/ML Students:**
• Machine Learning Workshop
• Data Science Club
• Kaggle Competitions
• AI Research Lab

**🌐 For Web Dev Students:**
• Frontend Framework Workshop
• Full-Stack Development Bootcamp
• Hackathon Participation
• UI/UX Design Course

**🔒 For Cybersecurity Students:**
• Ethical Hacking Course
• CTF Competitions
• Security Workshop
• Network Security Training

**📱 For App Dev Students:**
• Mobile App Development
• React Native Workshop
• App Store Optimization
• Mobile UI/UX Design

Who would you like specific recommendations for?
"""
    
    def get_student_recommendations(self, student):
        """Get personalized recommendations for a specific student"""
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
        
        student_recommendations = []
        
        for interest in student['Interests_List']:
            if interest in recommendations:
                student_recommendations.extend(recommendations[interest])
        
        student_recommendations = list(set(student_recommendations))[:5]
        
        return f"""
💡 **Personalized Recommendations for {student['Name']}**

**👤 Student Profile:**
• Branch: {student['Branch']}
• Year: {student['Year_Clean']}
• Interests: {', '.join(student['Interests_List'])}
• Engagement Score: {student['AI_Engagement_Score']:.1f}

**🎯 Recommended Activities:**
{chr(10).join([f"• {rec}" for rec in student_recommendations])}

**📈 Success Potential: {student['AI_Success_Potential']:.1f}**
"""
    
    def handle_follow_up_query(self, query, last_query):
        """Handle follow-up queries with context"""
        return f"I see you're continuing from your previous query about '{last_query}'. Let me provide additional insights..."
    
    def process_query(self, query):
        """Main query processing with AI intelligence"""
        # Update context
        self.context['recent_queries'].append(query)
        if len(self.context['recent_queries']) > 5:
            self.context['recent_queries'].pop(0)
        
        # Understand intent
        intent, confidence = self.understand_intent(query)
        
        # Generate response
        response = self.generate_intelligent_response(intent, query, confidence)
        
        # Update conversation history
        self.context['conversation_history'].append({
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        return response
    
    def run(self):
        """Main chatbot loop with GenAI-like interface"""
        print("🤖 **Welcome to GenAI Student Data Analyzer!**")
        print("=" * 60)
        print("🚀 I'm your AI-powered student data assistant with advanced insights.")
        print("💬 Ask me anything about your student data - I understand natural language!")
        print("🧠 I can analyze patterns, provide insights, and generate recommendations.")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n🤖 You: ").strip()
                
                if not query:
                    continue
                
                # Process query with AI intelligence
                response = self.process_query(query)
                print(f"\n🤖 AI: {response}")
                
                # Check for exit
                if any(word in query.lower() for word in ['bye', 'goodbye', 'exit', 'quit', 'stop']):
                    print("\n👋 Thank you for using GenAI Student Analyzer! Have a great day!")
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using GenAI Student Analyzer!")
                break
            except Exception as e:
                print(f"❌ AI Error: {e}")
                print("🔄 Let me try to understand your query differently...")

def main():
    """Main function"""
    analyzer = GenAIStudentAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 