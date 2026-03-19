import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import time
# --- CORRECTED PDF IMPORTS ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet # <--- CORRECTED LINE
import io
# -----------------------------

# --- CONFIGURATION ---
VIDEO_PATH = '/Users/abcd/Desktop/Sih project 2/Videos/9QJjM8y89B8xes1.MP4' 
CAP_SOURCE = VIDEO_PATH 
PDF_FILENAME = "Dance_Analysis_Report_" + time.strftime("%Y%m%d_%H%M%S") + ".pdf"

# --- MOCK CLASSIFIER DATA (as before) ---
MUDRA_IDEALS = {
    'Pataka': {'finger_tip_to_thumb_base_ratio': 0.8},
    'Tripataka': {'finger_tip_to_thumb_base_ratio': 0.5} 
}

# --- 1. HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates the angle in degrees between three 2D points (a, b, c) centered at b."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180.0 else 360 - angle

def mock_predict_mudra(landmarks):
    """Mocks the prediction of a trained ML model for the hand mudra."""
    if not landmarks:
        return "NO HAND", 0.0

    # Calculate key distances for feature extraction
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    index_tip = np.array([landmarks[8].x, landmarks[8].y])
    thumb_base = np.array([landmarks[2].x, landmarks[2].y])
    
    dist_tip_to_wrist = np.linalg.norm(index_tip - wrist)
    dist_thumb_base_to_wrist = np.linalg.norm(thumb_base - wrist)
    
    if dist_thumb_base_to_wrist == 0:
        return "ERR_NORM", 0.0

    feature_ratio = dist_tip_to_wrist / dist_thumb_base_to_wrist
    
    # Simple Rule-Based Check (Simulating a trained model's decision)
    if feature_ratio > 4.5:
        score = 0.90
        improvement = "Check finger extensions." if score < 0.95 else ""
        return f"Pataka {improvement}".strip(), score
    elif 2.0 < feature_ratio < 4.5:
        score = 0.75
        improvement = "Work on wrist alignment." if score < 0.8 else ""
        return f"Mushti {improvement}".strip(), score
    else:
        return "Unknown Mudra", 0.50

def mock_predict_pose(landmarks, mp_holistic):
    """Mocks the prediction of the overall body pose."""
    if not landmarks:
        return "NO POSE DETECTED", 0.0, 0.0

    # Pose Feature: Angle of the right knee (Hip-Knee-Ankle)
    hip_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].y]
    knee_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE].y]
    ankle_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE].y]
    
    knee_angle = calculate_angle(hip_r, knee_r, ankle_r)
    
    # Mock Feature 2: Torso Tilt Angle (Shoulder_Center to Hip_Center vs Vertical Axis)
    shoulder_c = [
        (landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x) / 2,
        (landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y) / 2
    ]
    hip_c = [
        (landmarks[mp_holistic.PoseLandmark.LEFT_HIP].x + landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x) / 2,
        (landmarks[mp_holistic.PoseLandmark.LEFT_HIP].y + landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].y) / 2
    ]
    vertical_ref = [hip_c[0], hip_c[1] - 0.1] 
    
    torso_tilt_angle = calculate_angle(shoulder_c, hip_c, vertical_ref)
    
    ideal_knee_angle = 120
    score = max(0, 1 - abs(knee_angle - ideal_knee_angle) / 50) 
    
    if score > 0.8:
        return f"Aramandi {int(knee_angle)}°", score, torso_tilt_angle
    elif knee_angle > 170:
        return f"Samapada {int(knee_angle)}°", score, torso_tilt_angle
    else:
        return f"Tricona {int(knee_angle)}°", score, torso_tilt_angle

# --- PDF GENERATION FUNCTION ---
def generate_pdf_report(plot_buffer, report_data):
    """Generates a PDF file containing the plots and the final text summary."""
    try:
        doc = SimpleDocTemplate(PDF_FILENAME, pagesize=letter)
        styles = getSampleStyleSheet() # <--- CORRECTED CALL
        story = []
    except ImportError:
        print("\n*** PDF GENERATION FAILED ***")
        print("Please ensure reportlab is installed.")
        return

    # Title and Video Info
    story.append(Paragraph("Dance Technique Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Analysis Time: {time.ctime()}", styles['Normal']))
    story.append(Paragraph(f"Video Source: {os.path.basename(CAP_SOURCE)}", styles['Normal']))
    story.append(Spacer(1, 18))

    # Image of Plots
    story.append(Paragraph("Time-Series Performance Graphs:", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    img = Image(plot_buffer)
    img.drawHeight = 350
    img.drawWidth = 500
    story.append(img)
    story.append(Spacer(1, 18))
    
    # Final Feedback
    story.append(Paragraph("Detailed Feedback Summary:", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    for line in report_data:
        style = styles['Code'] if "Recommendation" in line else styles['Normal']
        story.append(Paragraph(line.replace('**', ''), style))
        story.append(Spacer(1, 2))

    doc.build(story)
    print(f"\nPDF Report successfully generated and saved as: {PDF_FILENAME}")
    print("This file contains the full log and charts.")


# --- 2. MAIN ANALYSIS LOGIC ---
def run_dance_analysis():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAP_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source at {CAP_SOURCE}")
        return

    frame_num = 0
    analysis_log = []
    
    print("\n--- REAL-TIME CONSOLE FEEDBACK ---")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_num += 1
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape
        
        # --- PREDICTION ---
        mudra_name, mudra_score = mock_predict_mudra(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None)
        pose_name, pose_score, torso_tilt = mock_predict_pose(results.pose_landmarks.landmark if results.pose_landmarks else None, mp_holistic)

        # --- VISUALIZATION and LOGGING ---
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Torso Axis Calculation and Drawing
            hip_L = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP].x * w, landmarks[mp_holistic.PoseLandmark.LEFT_HIP].y * h]
            hip_R = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].y * h]
            shoulder_L = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * w, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * h]
            shoulder_R = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * h]
            
            hip_center = (int((hip_L[0] + hip_R[0]) / 2), int((hip_L[1] + hip_R[1]) / 2))
            shoulder_center = (int((shoulder_L[0] + shoulder_R[0]) / 2), int((shoulder_L[1] + shoulder_R[1]) / 2))
            
            cv2.line(image, (hip_center[0], 0), (hip_center[0], h), (0, 0, 255), 1) 
            cv2.line(image, hip_center, shoulder_center, (255, 0, 0), 2)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2, circle_radius=2))


        mudra_color = (0, 255, 0) if mudra_score > 0.75 else (0, 0, 255)
        pose_color = (0, 255, 0) if pose_score > 0.8 else (0, 0, 255)

        cv2.putText(image, f"Mudra: {mudra_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mudra_color, 2)
        cv2.putText(image, f"Pose: {pose_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pose_color, 2)
        cv2.putText(image, f"Torso Tilt: {torso_tilt:.1f} deg", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        analysis_log.append({
            'frame': frame_num,
            'mudra_score': mudra_score,
            'pose_score': pose_score,
            'torso_tilt': torso_tilt,
            'mudra_label': mudra_name.split(' ')[0]
        })
        
        print(f"Frame {frame_num:04d} | Mudra: {mudra_name:<25} | Score: {mudra_score:.2f} | Pose: {pose_name:<25} | Tilt: {torso_tilt:.1f}°")

        cv2.imshow('Dance Analysis System', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if analysis_log:
        df_log = pd.DataFrame(analysis_log)
        
        # --- Generate Plots and Save to Memory (No need to change) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(df_log['frame'], df_log['pose_score'], label='Overall Pose Score', color='blue')
        ax1.plot(df_log['frame'], df_log['mudra_score'], label='Hand Mudra Score', color='red')
        ax1.set_title('Technique Accuracy Over Time')
        ax1.set_ylabel('Correctness Score (0.0 to 1.0)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(df_log['frame'], df_log['torso_tilt'], label='Torso Tilt Angle (degrees)', color='purple')
        ax2.axhline(y=10, color='r', linestyle='--', label='Max Acceptable Tilt (10°)')
        ax2.set_title('Torso Stability and Axis Control')
        ax2.set_xlabel('Frame Number (Time)')
        ax2.set_ylabel('Torso Tilt from Vertical (degrees)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # --- Generate Final Feedback Report ---
        final_report_text = []
        low_pose_mean = df_log[df_log['pose_score'] < 0.75]
        low_tilt_frames = df_log[df_log['torso_tilt'] > 10]['frame']
        low_mudra_mean = df_log[df_log['mudra_score'] < 0.75]

        final_report_text.append("--- FINAL FEEDBACK REPORT ---")
        
        if not low_pose_mean.empty:
            final_report_text.append(f"BODY POSE: Found {len(low_pose_mean)} frames with weak body form (score < 0.75).")
        
        if not low_tilt_frames.empty:
            final_report_text.append(f"AXIS STABILITY (POWER): High instability in {len(low_tilt_frames)} frames (Torso tilt > 10°).")
            final_report_text.append("Recommendation: Work on core engagement to stabilize the Torso Axis.")
        else:
            final_report_text.append("BODY POSE: Overall body alignment and axis stability is excellent.")

        if not low_mudra_mean.empty:
            common_mudra = low_mudra_mean['mudra_label'].mode().iloc[0]
            final_report_text.append(f"HAND MUDRAS: Found {len(low_mudra_mean)} frames with weak mudra execution.")
            final_report_text.append(f"Recommendation: Review and refine the {common_mudra} mudra, focusing on precise finger positioning.")
        else:
            final_report_text.append("HAND MUDRAS: Mudra execution was strong and consistent.")
        
        print("\n" + "\n".join(final_report_text))
        
        # Call PDF generation function
        generate_pdf_report(buf, final_report_text)

    else:
        print("Analysis failed. Check video path or ensure a dancer is visible.")

if __name__ == "__main__":
    run_dance_analysis()