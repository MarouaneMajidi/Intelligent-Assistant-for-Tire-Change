import os
import sys
import time
import cv2
import numpy as np
import torch
import json
import threading
import queue
import tensorflow as tf
from ultralytics import YOLO
from collections import deque, Counter
from pathlib import Path

import re
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import vosk
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- MoViNet Integration ---
try:
    from official.projects.movinet.modeling import movinet
    from official.projects.movinet.modeling import movinet_model
except ImportError:
    print("="*60)
    print("WARNING: Could not import MoViNet components.")
    print("Action recognition will be disabled.")
    print("="*60)

class RealTimeActionRecognizer:
    def __init__(self, weights_path, config_path):
        self.weights_path = weights_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_id = self.config['model_id']
        self.num_classes = self.config['num_classes']
        self.resolution = self.config['resolution']
        self.class_names = self.config['class_names']
        
        # Initialize model and states
        self.model = None
        self.states = None
        self.prediction_history = deque(maxlen=15)
        self.load_model()
        
    def load_model(self):
        """Load and initialize the streaming model."""
        try:
            streaming_backbone = movinet.Movinet(
                model_id=self.model_id,
                causal=True,
                use_external_states=True
            )
            
            self.model = movinet_model.MovinetClassifier(
                backbone=streaming_backbone,
                num_classes=self.num_classes,
                output_states=True
            )
            
            # Build model
            self.model.build([1, 1, self.resolution, self.resolution, 3])
            self.model.load_weights(self.weights_path)
            print("✅ MoViNet model loaded successfully")
            self.reset_states()
            return True
        except Exception as e:
            print(f"❌ Error loading MoViNet: {e}")
            return False
        
    def reset_states(self):
        """Reset model states."""
        dummy_frame_shape = [1, 1, self.resolution, self.resolution, 3]
        self.states = self.model.init_states(tf.TensorShape(dummy_frame_shape))
        self.prediction_history.clear()
        print("MoViNet states reset")
        
    def format_frame(self, frame):
        """Format frame for model input."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.image.convert_image_dtype(frame_rgb, tf.float32)
        frame = tf.image.resize_with_pad(frame, self.resolution, self.resolution)
        return frame.numpy()
    
    def predict_frame(self, frame):
        """Predict action for a single formatted frame."""
        try:
            input_frame = frame[tf.newaxis, tf.newaxis, ...]
            logits, self.states = self.model.predict_on_batch((input_frame, self.states))
            
            probabilities = tf.nn.softmax(logits).numpy()[0]
            predicted_class_id = np.argmax(probabilities)
            confidence = probabilities[predicted_class_id]
            predicted_class_name = self.class_names[predicted_class_id]
            
            self.prediction_history.append(predicted_class_name)
            smoothed_prediction = Counter(self.prediction_history).most_common(1)[0][0]
            
            return {
                'class_name': predicted_class_name,
                'confidence': float(confidence),
                'smoothed_prediction': smoothed_prediction,
                'all_probabilities': probabilities.tolist()
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

# --- Application Constants ---
FLAT_TIRE_CLASS_NAME = "Flat_tire"
REQUIRED_TOOLS_CLASSES = {"Car_Jack", "Wheel_Wrench"}
VALIDATION_DURATION_SEC = 5.0
GRACE_PERIOD_DURATION_SEC = 2.5
TOOLS_CONFIDENCE_THRESHOLD = 0.6


# --- Intent Classifier Configuration ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # Update this path
TF_MODEL_DIR = "intent_model_bilstm_tf"
FAQ_FILE = "faq.json"

ACTION_VALIDATION_FRAMES = 15
ACTION_CONFIDENCE_THRESHOLD = 0.6
ACTION_CONFIDENCE_RESET_THRESHOLD = 0.3  # Reset when confidence drops below this

# --- Application States ---
STATE_DETECTING_FLAT_TIRE = "DETECTING_FLAT_TIRE"
STATE_COLLECTING_TOOLS = "COLLECTING_TOOLS"
STATE_ACTION_RECOGNITION = "ACTION_RECOGNITION"

# --- Action Steps ---
ACTION_STEPS = [
    "loosen_bolts",              # Step 1 - Loosen bolts before lifting the car
    "lift_car_with_jack",       # Step 2 - Lift the car using the jack
    "remove_bolts",             # Step 3 - Remove bolts completely
    "remove_tire",              # Step 4 - Take off the flat tire
    "place_spare_tire",         # Step 5 - Place the spare tire on the hub
    "hand_tighten_bolts",       # Step 6 - Start tightening bolts by hand
    "initial_wrench_tighten",   # Step 7 - Lightly tighten bolts with wrench while car is up
    "lower_car",                # Step 8 - Lower the car back to the ground
    "tighten_bolts"             # Step 9 - Fully tighten bolts after the car is lowered
]


# --- Display Offsets ---
Y_OFFSET_STEP_TITLE = 40
Y_OFFSET_MAIN_INSTRUCTION = 75
Y_OFFSET_STATUS_VALIDATION = 110
Y_OFFSET_CONFIRMATION_MSG = 110
Y_OFFSET_COLLECTED_LIST = 145
Y_OFFSET_NEEDED_LIST = 180
Y_OFFSET_ACTION_PROGRESS = 215
Y_OFFSET_ACTION_STEP = 250

def display_message(frame, message, y_offset=50, color=(255, 255, 0), 
                   font_scale=0.7, thickness=2, bg_color=(0, 0, 0)):
    """Displays a message on the frame with a background."""
    (text_width, text_height), baseline = cv2.getTextSize(
        message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    padding_x = 10
    padding_y = 5
    
    rect_x = 20 
    rect_y = y_offset - text_height - padding_y 
    rect_width = text_width + (2 * padding_x)
    rect_height = text_height + baseline + (2 * padding_y)

    if rect_y < 0:
        rect_y = 0
        y_offset = text_height + padding_y 

    if rect_x + rect_width > frame.shape[1] - 10:
        rect_width = frame.shape[1] - 10 - rect_x
        
    # Draw background
    cv2.rectangle(frame, (rect_x, rect_y), 
                 (rect_x + rect_width, rect_y + rect_height), 
                 bg_color, cv2.FILLED)
    
    # Draw text
    cv2.putText(frame, message, (rect_x + padding_x, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def inference_thread(recognizer, input_queue, output_queue):
    """Worker thread for running model inference."""
    while True:
        frame = input_queue.get()
        if frame is None:  # Termination signal
            break
            
        # Run prediction
        prediction = recognizer.predict_frame(frame)
        output_queue.put(prediction)

# --- Intent Classifier ---
class TTSThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.engine = pyttsx3.init()
        print("✅ TTS Engine initialized successfully.")

    def run(self):
        while self.running:
            try:
                text = self.queue.get(timeout=1.0)
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                pass

    def stop(self):
        self.running = False
        self.engine.stop()

class IntentClassifier:
    def __init__(self, tts_queue):
        self.tts_queue = tts_queue
        self.stopped = threading.Event()
        self.audio_queue = queue.Queue()
        
        # Load FAQ data
        try:
            with open(FAQ_FILE, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
            self.intent_to_answer_map = {item['intent']: item['answer'] for item in faq_data}
        except FileNotFoundError:
            print(f"Error: '{FAQ_FILE}' not found.")
            return

        # Initialize predictor
        try:
            self.predictor = self.load_predictor(TF_MODEL_DIR)
        except FileNotFoundError as e:
            print(f"Error initializing predictor: {e}")
            return

        # Initialize Vosk model
        try:
            self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            return
            
        self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
        
    def load_predictor(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at '{model_dir}'")
            
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        label_encoder_path = os.path.join(model_dir, 'label_encoder.json')
        with open(label_encoder_path, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(classes)

        model_path = os.path.join(model_dir, 'model.h5')
        model = load_model(model_path)
        max_len = model.input_shape[1]
        
        print("✅ TensorFlow Intent Predictor initialized")
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'max_len': max_len
        }

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_queue.put(bytes(indata))
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
        return text.strip()

    def predict_intent(self, text):
        predictor = self.predictor
        processed_text = self.preprocess_text(text)
        sequence = predictor['tokenizer'].texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=predictor['max_len'], padding='post', truncating='post')
        
        prediction = predictor['model'].predict(padded_sequence, verbose=0)
        predicted_idx = np.argmax(prediction, axis=1)[0]
        predicted_intent = predictor['label_encoder'].inverse_transform([predicted_idx])[0]
        return predicted_intent, prediction[0][predicted_idx]
        
    def start_listening(self):
        print("\n🎤 Assistant is now listening...")
        self.tts_queue.put("Hello! I'm your tire change assistant. How can I help?")
        
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self.audio_callback):
            while not self.stopped.is_set():
                try:
                    data = self.audio_queue.get(timeout=1.0)
                    if self.recognizer.AcceptWaveform(data):
                        result = self.recognizer.Result()
                        text = json.loads(result).get("text", "")
                        
                        if text and text.strip() != "":
                            print(f"\nRecognized: '{text}'")
                            
                            intent, probability = self.predict_intent(text)
                            answer = self.intent_to_answer_map.get(intent, 
                                          "I'm not sure how to answer that. Please ask about tire changing steps.")
                            
                            print(f"   -> Intent: '{intent}' ({probability*100:.2f}% confidence)")
                            print(f"   -> Response: {answer}")
                            
                            # Send response to TTS
                            self.tts_queue.put(answer)
                            
                except queue.Empty:
                    pass
                    
    def stop(self):
        self.stopped.set()

def main_orchestrator():
    """Main function to run the integrated tire change assistant."""
    # --- GPU Verification ---
    print("--- Verifying Processing Devices ---")
    # PyTorch GPU check
    if torch.cuda.is_available():
        print("✅ PyTorch: CUDA (GPU) available")
        torch_device = "cuda"
    else:
        print("⚠️ PyTorch: Using CPU")
        torch_device = "cpu"
    
    # TensorFlow GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ TensorFlow: GPU available")
        except RuntimeError as e:
            print(f"⚠️ TensorFlow GPU config error: {e}")
    else:
        print("⚠️ TensorFlow: No GPU found, using CPU")
    
    print("----------------------------------")
    
    # --- Model Paths ---
    flat_tire_model_path = r"flatV2.pt"
    tools_model_path = r"toolsV2.pt"
    action_weights_path = r'Streaming\streaming_movinet_weights.h5'
    action_config_path = r'Streaming\streaming_model_config.json'
    
    # Verify model paths
    model_paths = [
        flat_tire_model_path,
        tools_model_path,
        action_weights_path,
        action_config_path
    ]
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"ERROR: Model path not found: {path}")
            sys.exit(1)
    
    # --- Input Source Selection ---
    print("\n🎯 Choose Input Source:")
    print("1. Webcam (local camera)")
    print("2. Video File")
    print("3. IP Camera (network stream)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    img_source = None
    if choice == '1':
        img_source = 0
        print("INFO: Selected Webcam.")
    elif choice == '2':
        video_path = input("Enter video file path: ").strip().replace('"', '')
        if os.path.exists(video_path):
            img_source = video_path
            print(f"INFO: Selected Video File: {video_path}")
        else:
            print(f"ERROR: Video file not found: {video_path}")
            sys.exit(1)
    elif choice == '3':
        ip_url = input("Enter IP Camera URL (e.g., http://192.168.43.1:8080/video): ").strip()
        img_source = ip_url or "http://192.168.43.1:8080/video"
        print(f"INFO: Selected IP Camera: {img_source}")
    else:
        print("ERROR: Invalid choice")
        sys.exit(1)
    
    # --- Video Capture Setup ---
    print(f"\nINFO: Opening video source: {img_source}")
    cap = cv2.VideoCapture(img_source)
    if not cap.isOpened():
        print(f'ERROR: Unable to open video source: {img_source}')
        sys.exit(1)
    
    # Set resolution (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"INFO: Resolution set to: {actual_width}x{actual_height}")
    
    # --- State Machine Initialization ---
    app_state = STATE_DETECTING_FLAT_TIRE
    print(f"INFO: Initial state: {app_state}")
    
    # Load initial model (flat tire detection)
    model = YOLO(flat_tire_model_path, task='detect')
    labels = model.names
    
    # --- State Variables ---
    # Flat tire detection
    flat_tire_validation_start_time = None
    flat_tire_lost_temporarily_time = None
    
    # Tool collection (with simultaneous validation)
    confirmed_tools = set()
    tool_validation_timers = {tool: None for tool in REQUIRED_TOOLS_CLASSES}
    tool_lost_timers = {tool: None for tool in REQUIRED_TOOLS_CLASSES}
    tool_validation_status = {tool: False for tool in REQUIRED_TOOLS_CLASSES}
    
    # Action recognition
    action_recognizer = None
    movinet_input_queue = None
    movinet_output_queue = None
    movinet_thread = None
    current_action_step = 0
    action_validation_count = 0
    action_validation_start_time = None
    last_prediction_time = time.time()
    low_confidence_frames = 0

    # --- Initialize Intent Classifier ---
    try:
        tts_thread = TTSThread()
        tts_thread.start()
        
        intent_classifier = IntentClassifier(tts_thread.queue)
        # Start intent classifier in a separate thread
        intent_thread = threading.Thread(target=intent_classifier.start_listening)
        intent_thread.daemon = True
        intent_thread.start()
        print("✅ Intent classifier initialized and listening")
    except Exception as e:
        print(f"⚠️ Failed to initialize intent classifier: {e}")
        intent_classifier = None
        tts_thread = None
    
    # --- Performance Tracking ---
    frame_rate_buffer = deque(maxlen=30)
    avg_frame_rate = 0
    
    # --- Main Processing Loop ---
    try:
        while True:
            loop_start_time = time.perf_counter()
            
            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print('INFO: End of video stream')
                break
                
            # Resize if needed
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))
            
            # --- STATE: DETECTING_FLAT_TIRE ---
            if app_state == STATE_DETECTING_FLAT_TIRE:
                display_message(frame, "STEP 1: Find the Flat Tire", 
                               Y_OFFSET_STEP_TITLE, color=(200, 200, 200))
                
                # Run detection
                results = model(frame, verbose=False, conf=TOOLS_CONFIDENCE_THRESHOLD, device=torch_device)
                current_detections = results[0].boxes if results and results[0].boxes else []
                
                found_flat_tire = False
                for det in current_detections:
                    class_id = int(det.cls.item())
                    class_name = labels.get(class_id, f"ID:{class_id}")
                    
                    if class_name == FLAT_TIRE_CLASS_NAME:
                        found_flat_tire = True
                        conf = det.conf.item()
                        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), 
                                     (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name}: {conf:.2f}', 
                                   (xyxy[0], xyxy[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break
                
                if found_flat_tire:
                    # Reset lost timer if re-acquired
                    if flat_tire_lost_temporarily_time is not None:
                        flat_tire_lost_temporarily_time = None
                    
                    # Start validation timer if not already running
                    if flat_tire_validation_start_time is None:
                        flat_tire_validation_start_time = time.time()
                    
                    # Check validation duration
                    elapsed = time.time() - flat_tire_validation_start_time
                    if elapsed >= VALIDATION_DURATION_SEC:
                        display_message(frame, "Flat Tire Confirmed!", 
                                       Y_OFFSET_STATUS_VALIDATION, color=(0, 255, 0))
                        cv2.imshow('Tire Assistant', frame)
                        cv2.waitKey(1500)
                        
                        # Transition to next state
                        app_state = STATE_COLLECTING_TOOLS
                        confirmed_tools.clear()
                        model = YOLO(tools_model_path, task='detect')
                        labels = model.names
                        print("INFO: Transitioned to COLLECTING_TOOLS state")
                        
                        # Reset tool timers
                        for tool in REQUIRED_TOOLS_CLASSES:
                            tool_validation_timers[tool] = None
                            tool_lost_timers[tool] = None
                    else:
                        display_message(frame, 
                                       f"Confirming Flat Tire... {VALIDATION_DURATION_SEC - elapsed:.1f}s", 
                                       Y_OFFSET_STATUS_VALIDATION, color=(0, 255, 255))
                else:
                    # Handle temporary loss
                    if flat_tire_validation_start_time is not None:
                        if flat_tire_lost_temporarily_time is None:
                            flat_tire_lost_temporarily_time = time.time()
                        
                        time_since_lost = time.time() - flat_tire_lost_temporarily_time
                        if time_since_lost < GRACE_PERIOD_DURATION_SEC:
                            display_message(frame, 
                                           f"Searching... {GRACE_PERIOD_DURATION_SEC - time_since_lost:.1f}s", 
                                           Y_OFFSET_STATUS_VALIDATION, color=(255, 200, 0))
                        else:
                            flat_tire_validation_start_time = None
                            flat_tire_lost_temporarily_time = None
                            display_message(frame, "Point camera at the tire", 
                                           Y_OFFSET_MAIN_INSTRUCTION)
                    else:
                        display_message(frame, "Point camera at the tire", 
                                       Y_OFFSET_MAIN_INSTRUCTION)
            
            # --- STATE: COLLECTING_TOOLS (SIMULTANEOUS VALIDATION) ---
            elif app_state == STATE_COLLECTING_TOOLS:
                display_message(frame, "STEP 2: Collect Required Tools", 
                               Y_OFFSET_STEP_TITLE, color=(200, 200, 200))
                
                # Run detection
                results = model(frame, verbose=False, conf=TOOLS_CONFIDENCE_THRESHOLD, device=torch_device)
                current_yolo_detections = results[0].boxes if results and results[0].boxes else []
                
                detected_tools_in_frame_names = set()
                for det in current_yolo_detections:
                    classidx = int(det.cls.item())
                    classname = labels.get(classidx, f"ID:{classidx}")
                    if classname in REQUIRED_TOOLS_CLASSES:
                        detected_tools_in_frame_names.add(classname)
                        conf = det.conf.item()
                        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                        
                        # Draw bounding box
                        color_idx = classidx % 5
                        color = (31, 119, 180) if color_idx == 0 else \
                                (255, 127, 14) if color_idx == 1 else \
                                (44, 160, 44) if color_idx == 2 else \
                                (214, 39, 40) if color_idx == 3 else \
                                (148, 103, 189)
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                        cv2.putText(frame, f'{classname}:{conf:.2f}', (xyxy[0], xyxy[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Check if all tools are collected
                if confirmed_tools == REQUIRED_TOOLS_CLASSES:
                    app_state = STATE_ACTION_RECOGNITION
                    print("INFO: Transitioned to ACTION_RECOGNITION state")
                    continue
                
                # Simultaneous validation for all tools
                status_messages = []
                for tool in REQUIRED_TOOLS_CLASSES:
                    if tool in confirmed_tools:
                        continue  # Skip already confirmed tools
                        
                    if tool in detected_tools_in_frame_names:
                        # Tool detected in this frame
                        if tool_validation_timers[tool] is None:
                            # Start validation timer
                            tool_validation_timers[tool] = time.time()
                            tool_lost_timers[tool] = None
                            status_messages.append(f"Found {tool}, validating...")
                        else:
                            # Continue validation
                            elapsed = time.time() - tool_validation_timers[tool]
                            if elapsed >= VALIDATION_DURATION_SEC:
                                # Tool confirmed
                                confirmed_tools.add(tool)
                                tool_validation_timers[tool] = None
                                display_message(frame, f"{tool} Confirmed!", 
                                               Y_OFFSET_CONFIRMATION_MSG, color=(0, 255, 0))
                                cv2.imshow('Tire Assistant', frame)
                                cv2.waitKey(1000)
                            else:
                                status_messages.append(f"Validating {tool}: {VALIDATION_DURATION_SEC - elapsed:.1f}s")
                    else:
                        # Tool not detected
                        if tool_validation_timers[tool] is not None:
                            if tool_lost_timers[tool] is None:
                                # Start lost timer
                                tool_lost_timers[tool] = time.time()
                            else:
                                # Check grace period
                                time_since_lost = time.time() - tool_lost_timers[tool]
                                if time_since_lost >= GRACE_PERIOD_DURATION_SEC:
                                    # Tool lost, reset validation
                                    tool_validation_timers[tool] = None
                                    tool_lost_timers[tool] = None
                                    status_messages.append(f"{tool} validation reset")
                                else:
                                    status_messages.append(f"Searching {tool}: {GRACE_PERIOD_DURATION_SEC - time_since_lost:.1f}s")
                
                # Display status messages
                if status_messages:
                    display_message(frame, " | ".join(status_messages), 
                                   Y_OFFSET_STATUS_VALIDATION, color=(0, 255, 255))
                
                # Display collected and needed tools
                still_needed_tools = REQUIRED_TOOLS_CLASSES - confirmed_tools
                if confirmed_tools:
                    display_message(frame, f"Collected: {', '.join(sorted(list(confirmed_tools)))}", 
                                   Y_OFFSET_COLLECTED_LIST, color=(100, 200, 100))
                
                if still_needed_tools:
                    display_message(frame, f"Needed: {', '.join(sorted(list(still_needed_tools)))}", 
                                   Y_OFFSET_NEEDED_LIST, color=(255, 180, 0))
            
            # --- STATE: ACTION_RECOGNITION ---
            elif app_state == STATE_ACTION_RECOGNITION:
                # Initialize action recognition on first entry
                if action_recognizer is None:
                    print("INFO: Initializing action recognition system")
                    action_recognizer = RealTimeActionRecognizer(action_weights_path, action_config_path)
                    
                    # Setup async processing
                    movinet_input_queue = queue.Queue(maxsize=1)
                    movinet_output_queue = queue.Queue(maxsize=1)
                    movinet_thread = threading.Thread(
                        target=inference_thread,
                        args=(action_recognizer, movinet_input_queue, movinet_output_queue)
                    )
                    movinet_thread.daemon = True
                    movinet_thread.start()
                    
                    current_action_step = 0
                    action_validation_count = 0
                    action_validation_start_time = time.time()
                    last_prediction_confidence = 0.0
                    print(f"Starting action validation for: {ACTION_STEPS[current_action_step]}")
                
                # Display current step
                step_title = f"STEP {current_action_step+3}: {ACTION_STEPS[current_action_step]}"
                display_message(frame, step_title, Y_OFFSET_STEP_TITLE, color=(200, 200, 255))
                
                # Format and submit frame for processing
                formatted_frame = action_recognizer.format_frame(frame)
                if movinet_input_queue.empty():
                    try:
                        movinet_input_queue.put_nowait(formatted_frame)
                    except queue.Full:
                        pass
                
                # Get latest prediction
                prediction = None
                try:
                    prediction = movinet_output_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Process prediction
                if prediction:
                    current_confidence = prediction['confidence']
                    current_action = prediction['smoothed_prediction']
                    
                    # Check for significant confidence drop (≥0.05) for the current action
                    if (last_prediction_confidence - current_confidence >= 0.05 and
                        current_action == ACTION_STEPS[current_action_step]):
                        action_recognizer.reset_states()
                        print(f"Reset MoViNet: Confidence dropped by {last_prediction_confidence - current_confidence:.2f} for {current_action}")
                    
                    # Track confidence for next frame
                    last_prediction_confidence = current_confidence
                    
                    # Display prediction info
                    pred_text = f"Action: {current_action} ({current_confidence:.2f})"
                    display_message(frame, pred_text, Y_OFFSET_MAIN_INSTRUCTION, 
                                color=(255, 255, 0) if current_action == ACTION_STEPS[current_action_step] else (255, 100, 100))
                    
                    # Validate current step
                    if current_action == ACTION_STEPS[current_action_step] and current_confidence >= ACTION_CONFIDENCE_THRESHOLD:
                        action_validation_count += 1
                        
                        # Check if step is completed
                        if action_validation_count >= ACTION_VALIDATION_FRAMES:
                            display_message(frame, "Step Completed!", 
                                        Y_OFFSET_CONFIRMATION_MSG, color=(0, 255, 0))
                            cv2.imshow('Tire Assistant', frame)
                            cv2.waitKey(1500)
                            
                            # Move to next step
                            current_action_step += 1
                            action_validation_count = 0
                            
                            # Check if all steps are completed
                            if current_action_step >= len(ACTION_STEPS):
                                display_message(frame, "ALL STEPS COMPLETED!", 
                                            Y_OFFSET_MAIN_INSTRUCTION, 
                                            color=(0, 255, 0), font_scale=1.0)
                                cv2.imshow('Tire Assistant', frame)
                                cv2.waitKey(3000)
                                break
                            
                            # Reset states for new action
                            action_recognizer.reset_states()
                            last_prediction_confidence = 0.0
                            print(f"Starting action validation for: {ACTION_STEPS[current_action_step]}")
                    else:
                        action_validation_count = max(0, action_validation_count - 1)
                
                # Display validation progress
                progress_text = f"Validation: {action_validation_count}/{ACTION_VALIDATION_FRAMES}"
                display_message(frame, progress_text, Y_OFFSET_ACTION_PROGRESS, 
                            color=(0, 255, 255) if action_validation_count > 0 else (255, 165, 0))
                
            
            # --- FPS Calculation and Display ---
            loop_time = time.perf_counter() - loop_start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0
            frame_rate_buffer.append(fps)
            avg_frame_rate = sum(frame_rate_buffer) / len(frame_rate_buffer) if frame_rate_buffer else 0
            
            # Display FPS
            fps_color = (0, 255, 0) if avg_frame_rate > 15 else (0, 255, 255) if avg_frame_rate > 5 else (0, 0, 255)
            cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}', 
                       (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # Display frame
            cv2.imshow('Tire Assistant', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("INFO: Quitting...")
                break
            elif key == ord('p'):
                cv2.waitKey(-1)  # Pause
            elif key == ord('r') and app_state == STATE_ACTION_RECOGNITION:
                # Reset action recognition states
                if action_recognizer:
                    action_recognizer.reset_states()
                    action_validation_count = 0
                    print("Action recognition states reset")
    
    except KeyboardInterrupt:
        print("INFO: Process interrupted")
    finally:
        # Cleanup
        print("INFO: Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()


        
        # Stop MoViNet thread if running
        if movinet_thread and movinet_thread.is_alive():
            movinet_input_queue.put(None)
            movinet_thread.join(timeout=2.0)


        intent_classifier.stop()
        tts_thread.stop()
        
        print(f"INFO: Average FPS: {avg_frame_rate:.1f}")
        print("INFO: Program terminated")

if __name__ == '__main__':
    main_orchestrator()