import cv2
from fer import FER
from deepface import DeepFace
import threading
import queue
import time
import random


# Initialize variables
emotion = "Analyzing..."
current_tip = "Waiting for emotion detection..."
frame_count = 2 # Determine how often to analyze frames
analyze_every_n_frames = 15 # Calculates the most facial expression every 30 frames
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # Adjust this value based on your camera's capabilities


# Create a queue for frame processing
frame_queue = queue.Queue(maxsize=120)


# Initialize the FER model
fer_model = FER(mtcnn=True)


# Created list of phrases for each emotion
indicator = {
    "happy": ["Keep up the smiles!", "Great job with being happy, keep it going", "You're feeling good, wanna share your happiness with someone?", "Glad to see you're doing good now!", "Wowzers!, someone's cheered up!"],
    "sad": ["Life's too short for being sad, cheer up!", "Am I seeing a frown, how about flipping that to a smile :)", "Try reviving your happiest memories", "Feeling down? Try talking to your loved ones, or your friends", "Go out, you deserve a break, and maybe a sweet treat!"],
    "angry": ["You seem a little tense, take a deep breath and relax", "Feeling angry? Maybe you're just hangry, Go have a snack", "Remember that every emotion is temporary, So take it easy!", "Anger can be powerful; use it as fuel to get things done positively."],
    "neutral": ["You seem calm. A perfect time for some reflection!", "Feeling neutral? It's a great moment to take on a small task or goal.", "Keeping steady, perfect for a quick break or something creative!", "Sometimes feeling neutral is just what we need to stay grounded.", "Your calmness is inspiring! How about tackling something you have been putting off?"],
    "fear": ["Looking uneasy? Take a few deep breaths.", "Feeling anxious? A small relaxation exercise could help.", "It's okay to feel fearful. Take things one step at a time.", "Focus on something grounding—it could help ease your mind.", "Remember, courage isn't the absence of fear; it's acting despite it."],
}


def update_tip(detected_emotion):
    global current_tip
    if detected_emotion in indicator:
        current_tip = random.choice(indicator[detected_emotion])
    else:
        current_tip = "No prompt available for this emotion."


def analyze_emotion_thread():
    global emotion
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        try:
            # FER analysis
            fer_result = fer_model.detect_emotions(frame)
            if fer_result:
                fer_emotions = fer_result[0]['emotions']
                fer_emotion = max(fer_emotions, key=fer_emotions.get)
               
            # DeepFace analysis
            deepface_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            deepface_emotion = deepface_result[0]['dominant_emotion']
           
            # Combine results (you can adjust this logic)
            if fer_emotion == deepface_emotion:
                emotion = fer_emotion
            else:
                emotion = random.choice([fer_emotion, deepface_emotion])
           
            print(f"Detected Emotion: {emotion}")
            update_tip(emotion)
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
        frame_queue.task_done()


# Start the emotion analysis thread
emotion_thread = threading.Thread(target=analyze_emotion_thread)
emotion_thread.daemon = True
emotion_thread.start()


print("Press 'q' to quit the program.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break


    frame_count += 1


    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    # If a face is detected and it's time to analyze
    if len(faces) > 0 and frame_count % analyze_every_n_frames == 0:
        if frame_queue.qsize() < frame_queue.maxsize:
            frame_queue.put(frame)


    # Draw rectangle around the face and display emotion
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    # Display the tip
    tip_lines = [current_tip[i:i+40] for i in range(0, len(current_tip), 40)]
    for i, line in enumerate(tip_lines):
        cv2.putText(frame, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    cv2.imshow('Facial Expression Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Add a small delay to reduce CPU usage
    time.sleep(0.1)


# Clean up
frame_queue.put(None)  # Signal the emotion thread to exit
cap.release()
cv2.destroyAllWindows()
emotion_thread.join(timeout=1)


print("Program ended.")