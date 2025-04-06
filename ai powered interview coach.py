import streamlit as st
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
import sounddevice as sd
import wavio
import re
import threading
import queue
import time
import os
import hashlib
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Database setup (using JSON file for simplicity)
USER_DB_FILE = "users.json"

def init_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)

def hash_password(password):
    """Create a SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password meets requirements"""
    # At least 8 characters, one uppercase, one number, one special character
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def validate_username(username):
    """Validate username meets requirements"""
    if len(username) < 8:
        return False, "Username must be at least 8 characters"
    return True, "Username is valid"

def user_exists(username):
    """Check if a user already exists"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    return username in users

def add_user(username, password):
    """Add a new user to the database"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    users[username] = hash_password(password)
    
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def authenticate_user(username, password):
    """Authenticate a user"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    if username in users and users[username] == hash_password(password):
        return True
    return False

# Enhanced Video frame transformer for webcam with posture analysis
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.posture_status = "Unknown"
        self.shoulder_slope = 0
        self.head_position = "Unknown"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_posture_check = time.time()
        self.check_interval = 1.0  # Check posture every second
        self.posture_history = []  # Store recent posture assessments
        self.slouch_counter = 0
        self.posture_feedback = "Analyzing your posture..."
    
    def detect_face_and_shoulders(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # Use the largest face detected
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Approximate shoulder positions based on face position
        face_center_x = x + w//2
        face_bottom_y = y + h
        
        # Estimate shoulder area (wider than face and below it)
        shoulder_y = face_bottom_y + h//2
        shoulder_width = w * 2.5
        left_shoulder_x = max(0, int(face_center_x - shoulder_width//2))
        right_shoulder_x = min(img.shape[1], int(face_center_x + shoulder_width//2))
        
        return (x, y, w, h), (left_shoulder_x, right_shoulder_x, shoulder_y)
    
    def analyze_posture(self, img):
        face_rect, shoulder_points = self.detect_face_and_shoulders(img)
        
        if face_rect is None or shoulder_points is None:
            return "Cannot detect face", 0, "Unknown", img
        
        x, y, w, h = face_rect
        left_shoulder_x, right_shoulder_x, shoulder_y = shoulder_points
        
        # Draw face rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw approximate shoulder line
        cv2.line(img, (left_shoulder_x, shoulder_y), (right_shoulder_x, shoulder_y), (0, 255, 0), 2)
        
        # Calculate slope of shoulder line (ideally should be close to 0 for good posture)
        # For now, we're using a horizontal line, so slope is 0
        shoulder_slope = 0
        
        # Analyze head position relative to shoulders
        face_center_x = x + w//2
        shoulders_center_x = (left_shoulder_x + right_shoulder_x) // 2
        
        # Calculate horizontal offset between face center and shoulders center
        horizontal_offset = abs(face_center_x - shoulders_center_x)
        
        # Determine head position
        if horizontal_offset < w * 0.2:  # If offset is small
            head_position = "Centered"
        else:
            head_position = "Tilted" if face_center_x < shoulders_center_x else "Leaning"
        
        # Determine overall posture
        if head_position == "Centered" and abs(shoulder_slope) < 0.1:
            posture_status = "Good"
            self.slouch_counter = max(0, self.slouch_counter - 1)
        else:
            posture_status = "Needs Improvement"
            self.slouch_counter += 1
        
        # Add posture assessment to history
        self.posture_history.append(posture_status)
        if len(self.posture_history) > 10:  # Keep only last 10 assessments
            self.posture_history.pop(0)
        
        # Generate specific feedback
        if posture_status == "Good":
            self.posture_feedback = "Great posture! Keep it up!"
        elif head_position != "Centered":
            self.posture_feedback = "Try to keep your head centered above your shoulders."
        else:
            self.posture_feedback = "Straighten your shoulders for better posture."
        
        # Count how many recent posture checks were "Needs Improvement"
        if self.slouch_counter > 5:
            self.posture_feedback = "You're slouching! Sit up straight for a professional appearance."
        
        return posture_status, shoulder_slope, head_position, img
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Only analyze posture every check_interval seconds to reduce processing load
        current_time = time.time()
        if current_time - self.last_posture_check > self.check_interval:
            self.posture_status, self.shoulder_slope, self.head_position, img = self.analyze_posture(img)
            self.last_posture_check = current_time
        
        # Add interview tips and posture feedback at the top of the video
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create a semi-transparent overlay for text background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.7  # Transparency factor
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add interview tip
        tip_text = "Maintain eye contact and show confidence!"
        cv2.putText(img, tip_text, (10, 25), font, 0.6, (0, 255, 0), 2)
        
        # Add posture feedback with color based on status
        if self.posture_status == "Good":
            color = (0, 255, 0)  # Green for good posture
        else:
            color = (0, 165, 255)  # Orange for needs improvement
            
        cv2.putText(img, f"Posture: {self.posture_status} - {self.posture_feedback}", 
                   (10, 60), font, 0.6, color, 2)
        
        return img

class InterviewCoach:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(stopwords.words('english'))

        self.professional_words = {
            'accomplished', 'achieved', 'analyzed', 'coordinated', 'created',
            'delivered', 'developed', 'enhanced', 'executed', 'improved',
            'initiated', 'launched', 'managed', 'optimized', 'organized',
            'planned', 'resolved', 'spearheaded', 'streamlined', 'success'
        }

        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'so', 'well', 'just', 'stuff', 'things'
        }

        self.sample_rate = 44100  # Hz
        self.duration = 30  # seconds
        self.recording_in_progress = False
        self.recording_thread = None
        self.stop_recording = False

    def record_voice(self, filename="interview_response.wav"):
        """Record audio"""
        self.recording_in_progress = True
        self.stop_recording = False
        
        print(f"Recording your answer for {self.duration} seconds...")
        print("Speak now!")
        
        # Calculate total samples
        total_samples = int(self.duration * self.sample_rate)
        
        # Create array to store audio data
        recording = np.zeros((total_samples, 1))
        
        # Record in chunks to allow for stopping
        chunk_size = int(0.1 * self.sample_rate)  # 0.1 second chunks
        recorded_samples = 0
        
        # Start recording
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
        stream.start()
        
        while recorded_samples < total_samples and not self.stop_recording:
            # Calculate remaining samples in this chunk
            samples_to_record = min(chunk_size, total_samples - recorded_samples)
            
            # Record chunk
            data, overflowed = stream.read(samples_to_record)
            
            # Store in our recording array
            recording[recorded_samples:recorded_samples + len(data)] = data
            
            # Update position
            recorded_samples += len(data)
            
        # Stop and close the stream
        stream.stop()
        stream.close()
        
        # If we have any recorded data and not just stopped immediately
        if recorded_samples > self.sample_rate * 0.5:  # At least half a second
            # Trim recording to actual length
            recording = recording[:recorded_samples]
            
            # Save recording
            wavio.write(filename, recording, self.sample_rate, sampwidth=2)
            print(f"Recording saved to {filename}")
            return filename
        else:
            print("Recording canceled or too short")
            return None

    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        if audio_file is None:
            return ""
            
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Transcription complete!")
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
                return ""

    def analyze_tone(self, text):
        if not text:
            return {
                'score': 0,
                'sentiment': 'neutral',
                'feedback': 'No speech detected to analyze tone.'
            }

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        feedback = ""
        if sentiment == 'positive':
            feedback = "Your tone is positive and enthusiastic, which is great for an interview. Keep up the energy!"
            if compound_score > 0.5:
                feedback += " However, be careful not to come across as overly enthusiastic as it might seem insincere."
        elif sentiment == 'negative':
            feedback = "Your tone comes across as somewhat negative. Try to use more positive language and emphasize your strengths and achievements."
        else:
            feedback = "Your tone is neutral. While this is professional, try to inject some enthusiasm when discussing your achievements or interest in the role."

        return {
            'score': compound_score,
            'sentiment': sentiment,
            'feedback': feedback
        }

    def analyze_word_choice(self, text):
        if not text:
            return {
                'professional_word_count': 0,
                'filler_word_count': 0,
                'professional_words_used': [],
                'filler_words_used': [],
                'feedback': 'No speech detected to analyze word choice.'
            }

        words = nltk.word_tokenize(text.lower())
        professional_words_used = [word for word in words if word in self.professional_words]
        filler_words_used = [filler for filler in self.filler_words if filler in text.lower()]

        feedback = ""
        if professional_words_used:
            feedback += f"Good use of professional language! Words like {', '.join(professional_words_used[:3])} strengthen your responses. "
        else:
            feedback += "Consider incorporating more professional language to highlight your skills and achievements. "

        if filler_words_used:
            feedback += f"Try to reduce filler words/phrases like {', '.join(filler_words_used[:3])}. These can make you sound less confident."
        else:
            feedback += "You've done well avoiding filler words, which makes your speech sound more confident and prepared."

        return {
            'professional_word_count': len(professional_words_used),
            'filler_word_count': len(filler_words_used),
            'professional_words_used': professional_words_used,
            'filler_words_used': filler_words_used,
            'feedback': feedback
        }

    def analyze_confidence(self, text, tone_analysis):
        if not text:
            return {
                'confidence_score': 0,
                'feedback': 'No speech detected to analyze confidence.'
            }

        confidence_score = 5  # Base score out of 10
        sentiment_score = tone_analysis['score']
        if sentiment_score > 0:
            confidence_score += sentiment_score * 2
        elif sentiment_score < -0.2:
            confidence_score -= abs(sentiment_score) * 2

        hesitation_patterns = [
            r'\bI think\b', r'\bmaybe\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bI guess\b', r'\bsort of\b', r'\bkind of\b', r'\bI hope\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b'
        ]

        hesitation_count = sum(len(re.findall(pattern, text.lower())) for pattern in hesitation_patterns)
        confidence_score -= hesitation_count * 0.5

        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0

        if avg_sentence_length > 20:
            confidence_score += 1
        elif avg_sentence_length < 8:
            confidence_score -= 1

        confidence_score = max(0, min(10, confidence_score))

        if confidence_score >= 8:
            feedback = "You sound very confident. Your delivery is strong and assertive."
        elif confidence_score >= 6:
            feedback = "You sound reasonably confident. With a few adjustments, you could project even more authority."
        elif confidence_score >= 4:
            feedback = "Your confidence level seems moderate. Try speaking more assertively and avoiding hesitant language."
        else:
            feedback = "You may want to work on projecting more confidence. Try reducing hesitant phrases and speaking with more conviction."

        return {
            'confidence_score': confidence_score,
            'feedback': feedback
        }

    def provide_comprehensive_feedback(self, analysis_results):
        tone = analysis_results['tone']
        word_choice = analysis_results['word_choice']
        confidence = analysis_results['confidence']
        posture = analysis_results.get('posture', {'status': 'Not analyzed', 'feedback': 'No posture analysis available.'})

        feedback_text = "\n" + "=" * 50 + "\n"
        feedback_text += "INTERVIEW RESPONSE EVALUATION\n"
        feedback_text += "=" * 50 + "\n\n"

        feedback_text += "TONE ANALYSIS:\n"
        feedback_text += f"Sentiment: {tone['sentiment']} (Score: {tone['score']:.2f})\n"
        feedback_text += f"Feedback: {tone['feedback']}\n\n"

        feedback_text += "WORD CHOICE ANALYSIS:\n"
        feedback_text += f"Professional words used: {word_choice['professional_word_count']}\n"
        if word_choice['professional_words_used']:
            feedback_text += f"Examples: {', '.join(word_choice['professional_words_used'][:3])}\n"

        feedback_text += f"Filler words/phrases used: {word_choice['filler_word_count']}\n"
        if word_choice['filler_words_used']:
            feedback_text += f"Examples: {', '.join(word_choice['filler_words_used'][:3])}\n"

        feedback_text += f"Feedback: {word_choice['feedback']}\n\n"

        feedback_text += "CONFIDENCE ASSESSMENT:\n"
        feedback_text += f"Confidence Score: {confidence['confidence_score']:.1f}/10\n"
        feedback_text += f"Feedback: {confidence['feedback']}\n\n"

        # Include posture feedback if available
        if 'posture' in analysis_results:
            feedback_text += "POSTURE ASSESSMENT:\n"
            feedback_text += f"Status: {posture['status']}\n"
            feedback_text += f"Feedback: {posture['feedback']}\n\n"

        avg_score = (tone['score'] + 1) * 5 + confidence['confidence_score']
        avg_score /= 2

        if avg_score >= 8:
            feedback_text += "Excellent interview response! You presented yourself very well.\n"
        elif avg_score >= 6:
            feedback_text += "Good interview response. With some minor improvements, you'll make an even stronger impression.\n"
        elif avg_score >= 4:
            feedback_text += "Acceptable interview response. Focus on the improvement areas mentioned above.\n"
        else:
            feedback_text += "Your interview response needs improvement. Consider practicing more with the suggestions provided.\n"

        feedback_text += "\nAREAS TO FOCUS ON:\n"
        improvement_areas = []

        if tone['score'] < 0:
            improvement_areas.append("Using more positive language")
        if word_choice['filler_word_count'] > 3:
            improvement_areas.append("Reducing filler words/phrases")
        if word_choice['professional_word_count'] < 2:
            improvement_areas.append("Incorporating more professional vocabulary")
        if confidence['confidence_score'] < 5:
            improvement_areas.append("Building confidence in delivery")
        if 'posture' in analysis_results and posture['status'] != 'Good':
            improvement_areas.append("Improving your posture and body language")

        if improvement_areas:
            for i, area in enumerate(improvement_areas, 1):
                feedback_text += f"{i}. {area}\n"
        else:
            feedback_text += "Great job! Keep practicing to maintain your strong performance.\n"

        feedback_text += "=" * 50 + "\n"

        return feedback_text

    def analyze_text_input(self, text):
        tone_analysis = self.analyze_tone(text)
        word_choice_analysis = self.analyze_word_choice(text)
        confidence_analysis = self.analyze_confidence(text, tone_analysis)

        analysis_results = {
            'tone': tone_analysis,
            'word_choice': word_choice_analysis,
            'confidence': confidence_analysis,
            'text': text
        }

        return analysis_results

def create_login_page():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        if username and password:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

def create_signup_page():
    st.subheader("Create an Account")
    
    username = st.text_input("Username (minimum 8 characters)", key="signup_username")
    password = st.text_input("Password", type="password", help="Password must contain at least 8 characters, one uppercase letter, one number, and one special character", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.button("Sign Up"):
        if not username or not password or not confirm_password:
            st.warning("Please fill in all fields")
            return
            
        valid_username, username_msg = validate_username(username)
        if not valid_username:
            st.error(username_msg)
            return
            
        valid_password, password_msg = validate_password(password)
        if not valid_password:
            st.error(password_msg)
            return
            
        if password != confirm_password:
            st.error("Passwords do not match")
            return
            
        if user_exists(username):
            st.error("Username already exists. Please choose another one.")
            return
            
        add_user(username, password)
        st.success("Account created successfully! Please log in.")
        st.session_state.show_login = True
        st.rerun()

def main_app():
    st.title("AI-Powered Interview Coach")
    st.write(f"Welcome, {st.session_state.username}! Practice your interview skills and receive feedback on your responses.")
    
    # Create a sidebar with options
    menu = ["Practice Interview", "View Tips", "Settings", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Practice Interview":
        st.header("Practice Interview")
        
        # Initialize camera_active state if it doesn't exist
        if "camera_active" not in st.session_state:
            st.session_state.camera_active = True
        
        # Initialize posture_analysis state if it doesn't exist
        if "posture_analysis_active" not in st.session_state:
            st.session_state.posture_analysis_active = True
        
        # Webcam feedback section - only show when camera is active
        if st.session_state.camera_active:
            st.subheader("Video Feedback")
            
            # Add posture analysis toggle
            st.session_state.posture_analysis_active = st.checkbox(
                "Enable posture analysis", 
                value=st.session_state.posture_analysis_active
            )
            
            webrtc_ctx = webrtc_streamer(
                key="interview-webcam",
                video_transformer_factory=VideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
            )
            
            if webrtc_ctx.video_transformer:
                st.info("Webcam is active. You can now see yourself as you would appear in an interview.")
                
                if st.session_state.posture_analysis_active:
                    st.markdown("""
                    **Posture tips for interviews:**
                    - Sit up straight with shoulders back
                    - Keep your head centered and level
                    - Avoid slouching or leaning to one side
                    - Position your camera at eye level
                    """)
                
                st.markdown("""
                **Tips for good video presence:**
                - Maintain eye contact with the camera
                - Ensure your face is well-lit
                - Keep a neutral background
                - Use natural hand gestures when speaking
                """)
                
                # Add manual camera control
                if st.button("Stop Camera"):
                    st.session_state.camera_active = False
                    st.rerun()
        else:
            if st.button("Start Camera"):
                st.session_state.camera_active = True
                st.rerun()
        
        # Interview practice section
        st.subheader("Voice Analysis")
        coach = InterviewCoach()
        
        # Sample interview questions
        questions = [
            "Tell me about yourself.",
            "What are your greatest strengths?",
            "What is your biggest weakness?",
            "Why do you want to work at our company?",
            "Describe a challenging situation and how you handled it.",
            "Where do you see yourself in 5 years?",
            "Custom question (type below)"
        ]
        
        selected_question = st.selectbox("Select an interview question to answer:", questions)
        
        if selected_question == "Custom question (type below)":
            custom_question = st.text_input("Enter your custom question:")
            if custom_question:
                st.write(f"Question: {custom_question}")
        else:
            st.write(f"Question: {selected_question}")
            
        # Recording duration selection
        duration = st.selectbox("Select Recording Duration (seconds)", [15, 30, 45, 60, 90, 120], index=1)
        coach.duration = duration
        
        # Record button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Record Answer"):
                with st.spinner("Recording in progress..."):
                    audio_file = coach.record_voice()
                    if audio_file:
                        st.success("Recording complete! Transcribing...")
                        text = coach.transcribe_audio(audio_file)
                        if text:
                            st.session_state.transcribed_text = text
                            
                            # Collect posture data if available and posture analysis is active
                            posture_data = None
                            if st.session_state.camera_active and st.session_state.posture_analysis_active and webrtc_ctx and webrtc_ctx.video_transformer:
                                posture_data = {
                                    'status': webrtc_ctx.video_transformer.posture_status,
                                    'feedback': webrtc_ctx.video_transformer.posture_feedback
                                }
                            
                            # Perform voice analysis
                            analysis_results = coach.analyze_text_input(text)
                            
                            # Add posture analysis if available
                            if posture_data:
                                analysis_results['posture'] = posture_data
                                
                            st.session_state.analysis_results = analysis_results
                            
                            # Stop camera after analysis if auto-stop is enabled
                            if st.session_state.auto_stop_camera:
                                st.session_state.camera_active = False
                            st.rerun()
                        else:
                            st.error("Transcription failed. Please try again.")
                    else:
                        st.error("Recording was canceled or too short.")
        
        with col2:
            if st.button("Text Input Instead"):
                st.session_state.show_text_input = True
                st.rerun()
                
        if "show_text_input" in st.session_state and st.session_state.show_text_input:
            text_input = st.text_area("Type your answer instead:", height=150)
            if st.button("Analyze Text"):
                if text_input:
                    st.session_state.transcribed_text = text_input
                    
                    # Collect posture data if available and posture analysis is active
                    posture_data = None
                    if st.session_state.camera_active and st.session_state.posture_analysis_active and 'webrtc_ctx' in locals() and webrtc_ctx.video_transformer:
                        posture_data = {
                            'status': webrtc_ctx.video_transformer.posture_status,
                            'feedback': webrtc_ctx.video_transformer.posture_feedback
                        }
                    
                    # Perform voice analysis
                    analysis_results = coach.analyze_text_input(text_input)
                    
                    # Add posture analysis if available
                    if posture_data:
                        analysis_results['posture'] = posture_data
                        
                    st.session_state.analysis_results = analysis_results
                    st.session_state.show_text_input = False
                    
                    # Stop camera after analysis if auto-stop is enabled
                    if st.session_state.auto_stop_camera:
                        st.session_state.camera_active = False
                    st.rerun()
                else:
                    st.warning("Please enter some text to analyze.")
        
        # Display transcription and analysis if available
        if "transcribed_text" in st.session_state and "analysis_results" in st.session_state:
            st.subheader("Your Answer")
            st.write(st.session_state.transcribed_text)
            
            st.subheader("Analysis Results")
            feedback = coach.provide_comprehensive_feedback(st.session_state.analysis_results)
            st.markdown(feedback)
            
            # Create visualizations of the analysis
            st.subheader("Analysis Visualization")
            
            # Create columns for the visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment Analysis
                sentiment_score = st.session_state.analysis_results['tone']['score']
                sentiment_label = st.session_state.analysis_results['tone']['sentiment']
                
                st.subheader("Tone Analysis")
                st.progress((sentiment_score + 1) / 2)  # Map from [-1, 1] to [0, 1]
                st.write(f"Sentiment: {sentiment_label.capitalize()} ({sentiment_score:.2f})")
                
                # Word Choice Metrics
                st.subheader("Word Choice")
                prof_words = st.session_state.analysis_results['word_choice']['professional_word_count']
                filler_words = st.session_state.analysis_results['word_choice']['filler_word_count']
                
                # Simple bar chart for word counts
                st.bar_chart({
                    'Professional Words': [prof_words],
                    'Filler Words': [filler_words]
                })
            
            with col2:
                # Confidence Score
                confidence_score = st.session_state.analysis_results['confidence']['confidence_score']
                
                st.subheader("Confidence Score")
                st.progress(confidence_score / 10)  # Scale to [0, 1]
                st.write(f"Confidence: {confidence_score:.1f}/10")
                
                # Posture status if available
                if 'posture' in st.session_state.analysis_results:
                    st.subheader("Posture Status")
                    posture_status = st.session_state.analysis_results['posture']['status']
                    if posture_status == "Good":
                        st.success(posture_status)
                    else:
                        st.warning(posture_status)
            
            # Add share or save options
            st.subheader("Save or Share Results")
            if st.button("Save Analysis to File"):
                # Create a dictionary of results to save
                results_to_save = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "question": selected_question if selected_question != "Custom question (type below)" else custom_question,
                    "answer": st.session_state.transcribed_text,
                    "analysis": st.session_state.analysis_results,
                    "feedback": feedback
                }
                
                # Save to a json file named with timestamp
                filename = f"interview_analysis_{st.session_state.username}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w") as f:
                    json.dump(results_to_save, f, indent=4)
                
                st.success(f"Analysis saved to {filename}")
                
            # Option to clear results and start over
            if st.button("Clear and Start Over"):
                if "transcribed_text" in st.session_state:
                    del st.session_state.transcribed_text
                if "analysis_results" in st.session_state:
                    del st.session_state.analysis_results
                st.session_state.show_text_input = False
                st.rerun()

    elif choice == "View Tips":
        st.header("Interview Tips & Techniques")
        
        # Create tabs for different categories of tips
        tabs = st.tabs(["General Tips", "Body Language", "Voice & Speech", "Common Questions", "Industry Specific"])
        
        with tabs[0]:
            st.subheader("General Interview Tips")
            st.markdown("""
            ### Preparation
            - Research the company thoroughly
            - Prepare specific examples from your experience
            - Practice answering common questions
            - Prepare questions to ask the interviewer
            
            ### During the Interview
            - Arrive 10-15 minutes early
            - Bring extra copies of your resume
            - Listen carefully to questions before answering
            - Use the STAR method (Situation, Task, Action, Result) for behavioral questions
            - Be positive and enthusiastic
            
            ### Follow-up
            - Send a thank-you email within 24 hours
            - Reference specific topics from the interview
            - Express continued interest in the position
            """)
        
        with tabs[1]:
            st.subheader("Body Language Tips")
            st.markdown("""
            ### Posture
            - Sit upright with shoulders back
            - Avoid slouching or leaning to one side
            - Keep feet planted on the floor
            
            ### Hand Gestures
            - Use natural hand gestures when making important points
            - Avoid fidgeting, tapping, or playing with objects
            - Keep hands visible, not hidden under the table
            
            ### Facial Expressions
            - Make appropriate eye contact (70-80% of the time)
            - Smile naturally when greeting and during lighter moments
            - Show interest through nodding and expressions
            
            ### For Video Interviews
            - Position camera at eye level
            - Look at the camera, not at yourself on screen
            - Ensure proper lighting (light source in front of you)
            - Have a clean, professional background
            """)
        
        with tabs[2]:
            st.subheader("Voice & Speech Tips")
            st.markdown("""
            ### Volume & Pace
            - Speak clearly at a moderate pace
            - Vary your tone to emphasize important points
            - Avoid speaking too quickly when nervous
            
            ### Language
            - Use professional vocabulary relevant to the industry
            - Avoid filler words ("um", "like", "you know")
            - Use positive language and active verbs
            
            ### Confidence
            - Practice speaking with conviction
            - Avoid hedging phrases ("I think maybe", "sort of")
            - Take a brief pause before answering difficult questions
            
            ### Storytelling
            - Structure responses with a clear beginning, middle, and end
            - Be concise - aim for 1-2 minute responses
            - Highlight results and learnings in your stories
            """)
        
        with tabs[3]:
            st.subheader("Common Interview Questions")
            st.markdown("""
            ### About You
            - "Tell me about yourself"
            - "What are your greatest strengths/weaknesses?"
            - "Where do you see yourself in 5 years?"
            
            ### Experience
            - "Describe a challenging situation and how you handled it"
            - "Tell me about a project you're proud of"
            - "What's your biggest professional achievement?"
            
            ### About the Company
            - "Why do you want to work here?"
            - "What do you know about our company/products?"
            - "Why should we hire you?"
            
            ### Behavioral
            - "Tell me about a time you failed and what you learned"
            - "Describe how you work under pressure"
            - "Give an example of how you resolved a conflict"
            """)
        
        with tabs[4]:
            st.subheader("Industry-Specific Tips")
            
            industries = ["Technology", "Finance", "Healthcare", "Marketing", "Education"]
            selected_industry = st.selectbox("Select an industry", industries)
            
            if selected_industry == "Technology":
                st.markdown("""
                ### Technology Interview Tips
                - Be prepared to discuss specific technical skills listed in the job description
                - Expect technical questions or coding challenges
                - Show your problem-solving approach, not just the solution
                - Demonstrate continuous learning and staying current with technologies
                - Be ready to discuss previous projects in detail
                """)
            
            elif selected_industry == "Finance":
                st.markdown("""
                ### Finance Interview Tips
                - Be up-to-date on market trends and financial news
                - Prepare for case studies and financial modeling exercises
                - Demonstrate analytical thinking and attention to detail
                - Show understanding of risk management principles
                - Be ready to discuss regulatory knowledge relevant to the role
                """)
            
            elif selected_industry == "Healthcare":
                st.markdown("""
                ### Healthcare Interview Tips
                - Emphasize patient care and empathy
                - Demonstrate knowledge of relevant regulations (HIPAA, etc.)
                - Highlight experience with electronic health records systems
                - Show commitment to continuing education
                - Prepare examples of multidisciplinary collaboration
                """)
            
            elif selected_industry == "Marketing":
                st.markdown("""
                ### Marketing Interview Tips
                - Bring a portfolio of your work
                - Be prepared to discuss metrics and campaign results
                - Show knowledge of current digital marketing trends
                - Demonstrate creativity and strategic thinking
                - Prepare case studies of successful campaigns you've worked on
                """)
            
            elif selected_industry == "Education":
                st.markdown("""
                ### Education Interview Tips
                - Be ready to discuss teaching philosophy
                - Prepare examples of classroom management techniques
                - Demonstrate adaptability to different learning styles
                - Show knowledge of current educational technologies
                - Highlight experience with assessment methods
                """)

    elif choice == "Settings":
        st.header("Settings")
        
        st.subheader("Account Settings")
        if st.button("Change Password"):
            st.session_state.change_password = True
            
        if "change_password" in st.session_state and st.session_state.change_password:
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if not current_password or not new_password or not confirm_new_password:
                    st.warning("Please fill in all password fields")
                elif not authenticate_user(st.session_state.username, current_password):
                    st.error("Current password is incorrect")
                elif new_password != confirm_new_password:
                    st.error("New passwords do not match")
                else:
                    valid_password, password_msg = validate_password(new_password)
                    if not valid_password:
                        st.error(password_msg)
                    else:
                        # Update password in DB
                        with open(USER_DB_FILE, "r") as f:
                            users = json.load(f)
                        
                        users[st.session_state.username] = hash_password(new_password)
                        
                        with open(USER_DB_FILE, "w") as f:
                            json.dump(users, f)
                            
                        st.success("Password updated successfully!")
                        st.session_state.change_password = False
                        st.rerun()
        
        st.subheader("App Settings")
        
        # Camera settings
        st.checkbox("Auto-stop camera after analysis", 
                    value=st.session_state.get("auto_stop_camera", False),
                    key="auto_stop_camera")
        
        # Analysis settings
        st.checkbox("Include posture analysis in feedback", 
                    value=st.session_state.get("include_posture", True),
                    key="include_posture")
        
        # Save settings button
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
        
    elif choice == "Logout":
        if st.button("Confirm Logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Logged out successfully")
            st.rerun()

def main():
    st.set_page_config(
        page_title="AI Interview Coach",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize DB
    init_db()
    
    # Initialize session state for login status if it doesn't exist
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if "show_login" not in st.session_state:
        st.session_state.show_login = True
    
    if "auto_stop_camera" not in st.session_state:
        st.session_state.auto_stop_camera = False
    
    # Show app title
    st.title("AI Interview Coach")
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        st.write("Welcome to AI Interview Coach! Login or create an account to get started.")
        
        # Create tabs for login and signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            create_login_page()
        
        with tab2:
            create_signup_page()
    else:
        # Main application
        main_app()

if __name__ == "__main__":
    main()
