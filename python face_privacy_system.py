import cv2
import numpy as np
import streamlit as st
import time
import mediapipe as mp
import os


# === Combined Face Recognition and Evasion System ===

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection for recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}

    def load_training_data(self, dataset_path):
        """Load face images from dataset and train recognizer"""
        faces, labels, label_map = [], [], {}
        label_id = 0

        if not os.path.exists(dataset_path):
            st.warning(f"Dataset path not found: {dataset_path}. Using default recognizer.")
            return False

        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue
            label_map[label_id] = person_name
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(person_path, filename)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    face_rects = self.face_cascade.detectMultiScale(image, 1.1, 5)
                    for (x, y, w, h) in face_rects:
                        face_roi = image[y:y + h, x:x + w]
                        resized_face = cv2.resize(face_roi, (200, 200))
                        faces.append(resized_face)
                        labels.append(label_id)
            label_id += 1

        if len(faces) < 2:
            st.warning("Not enough training data. Using default recognizer.")
            return False

        try:
            self.recognizer.train(faces, np.array(labels))
            self.label_map = label_map
            st.success(f"✅ Model trained with {len(label_map)} people: {list(label_map.values())}")
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False

    def recognize_faces(self, frame):
        """Perform face recognition on frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

        recognition_results = []

        for (x, y, w, h) in faces_rect:
            face_roi = gray[y:y + h, x:x + w]

            if face_roi.size == 0:
                continue

            try:
                resized_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.recognizer.predict(resized_roi)

                if confidence < 60 and label in self.label_map:
                    name = self.label_map[label]
                    color = (0, 255, 0)  # Green for recognized
                    status = "recognized"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    status = "unknown"

                recognition_results.append({
                    'coords': (x, y, w, h),
                    'name': name,
                    'confidence': confidence,
                    'color': color,
                    'status': status
                })

            except Exception as e:
                continue

        return recognition_results


class StrongFaceEvader:
    def __init__(self):
        # Initialize face detection for evasion
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        """Detect faces using multiple methods"""
        faces = []

        # MediaPipe detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                faces.append((x, y, face_w, face_h))

        # OpenCV detection as backup
        if not faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_cv = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            faces = [(x, y, w, h) for (x, y, w, h) in faces_cv]

        return faces


class StrongEvasionTechniques:
    def __init__(self):
        self.techniques = {
            "neural_scrambler": "Strong neural-style disruption",
            "pixel_destroyer": "Aggressive pixel-level attacks",
            "feature_destroyer": "Destroys key facial features",
            "adversarial_attack": "Adversarial noise patterns",
            "color_chaos": "Extreme color manipulation",
            "triple_threat": "Combines all strong techniques"
        }

    def neural_scrambler(self, frame, faces):
        """Strong neural-style disruption that breaks deep learning models"""
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            # Strong high-frequency noise injection
            noise = np.random.randint(-25, 25, face_roi.shape, dtype=np.int16)
            face_roi = np.clip(face_roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Extreme color channel separation
            b, g, r = cv2.split(face_roi)
            b = np.roll(b, 3, axis=1)
            g = np.roll(g, -2, axis=0)
            r = np.roll(r, 2, axis=1)
            face_roi = cv2.merge([b, g, r])

            # Strong local histogram destruction
            for i in range(0, face_roi.shape[0], 10):
                for j in range(0, face_roi.shape[1], 10):
                    block = face_roi[i:i + 10, j:j + 10]
                    if block.size > 0:
                        # Random histogram equalization or inversion
                        if np.random.random() > 0.5:
                            # Convert to grayscale for equalization
                            gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                            gray_block = cv2.equalizeHist(gray_block)
                            block = cv2.cvtColor(gray_block, cv2.COLOR_GRAY2BGR)
                        else:
                            block = 255 - block
                        face_roi[i:i + 10, j:j + 10] = block

            frame[y:y + h, x:x + w] = face_roi
        return frame

    def pixel_destroyer(self, frame, faces):
        """Aggressive pixel-level attacks that break recognition"""
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            h, w = face_roi.shape[:2]

            # Method 1: Extreme noise patterns
            # Create multiple noise layers
            noise1 = np.random.randint(-40, 40, (h, w, 3), dtype=np.int16)
            noise2 = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)

            face_roi = np.clip(face_roi.astype(np.int16) + noise1 + noise2, 0, 255).astype(np.uint8)

            # Method 2: Block-wise scrambling
            block_size = 8
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if i + block_size <= h and j + block_size <= w:
                        block = face_roi[i:i + block_size, j:j + block_size]
                        # Randomly shuffle pixels within block
                        if np.random.random() > 0.7:
                            shuffled = block.reshape(-1, 3)
                            np.random.shuffle(shuffled)
                            face_roi[i:i + block_size, j:j + block_size] = shuffled.reshape(block_size, block_size, 3)

            # Method 3: Edge destruction with strong blur
            face_roi = cv2.GaussianBlur(face_roi, (7, 7), 3)

            # Method 4: Random pixel value inversion
            mask = np.random.random((h, w)) > 0.3
            face_roi[mask] = 255 - face_roi[mask]

            frame[y:y + h, x:x + w] = face_roi
        return frame

    def feature_destroyer(self, frame, faces):
        """Aggressively destroy key facial features"""
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            h, w = face_roi.shape[:2]

            # Define key facial regions more aggressively
            regions = [
                # Eyes - larger regions
                (int(h * 0.2), int(h * 0.45), int(w * 0.2), int(w * 0.4)),
                (int(h * 0.2), int(h * 0.45), int(w * 0.6), int(w * 0.8)),
                # Nose - larger region
                (int(h * 0.35), int(h * 0.65), int(w * 0.35), int(w * 0.65)),
                # Mouth - larger region
                (int(h * 0.55), int(h * 0.85), int(w * 0.25), int(w * 0.75)),
                # Cheeks
                (int(h * 0.3), int(h * 0.6), int(w * 0.1), int(w * 0.3)),
                (int(h * 0.3), int(h * 0.6), int(w * 0.7), int(w * 0.9))
            ]

            for y1, y2, x1, x2 in regions:
                if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                    region = face_roi[y1:y2, x1:x2]
                    if region.size > 0:
                        # Apply multiple destructive transformations
                        if np.random.random() > 0.3:
                            # Strong Gaussian noise
                            noise = np.random.randint(-40, 40, region.shape, dtype=np.int16)
                            region = np.clip(region.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        if np.random.random() > 0.5:
                            # Color inversion
                            region = 255 - region
                        if np.random.random() > 0.7:
                            # Extreme blur
                            region = cv2.GaussianBlur(region, (7, 7), 5)

                        face_roi[y1:y2, x1:x2] = region

            # Apply overall face transformations
            # Strong brightness/contrast variation
            alpha = np.random.uniform(0.5, 1.5)
            beta = np.random.randint(-30, 30)
            face_roi = cv2.convertScaleAbs(face_roi, alpha=alpha, beta=beta)

            frame[y:y + h, x:x + w] = face_roi
        return frame

    def adversarial_attack(self, frame, faces):
        """Apply adversarial noise patterns that break neural networks"""
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            # Create adversarial patterns
            h, w = face_roi.shape[:2]

            # Pattern 1: High-frequency grid noise
            for i in range(0, h, 4):
                for j in range(0, w, 4):
                    if i + 2 <= h and j + 2 <= w:
                        face_roi[i:i + 2, j:j + 2] = np.random.randint(0, 255, (2, 2, 3))

            # Pattern 2: Sinusoidal adversarial noise
            x_coords = np.arange(w).reshape(1, -1)
            y_coords = np.arange(h).reshape(-1, 1)

            # Multiple frequency noise patterns
            for freq in [5, 13, 21]:
                pattern = 30 * (np.sin(2 * np.pi * freq * x_coords / w) *
                                np.sin(2 * np.pi * freq * y_coords / h))
                pattern = np.stack([pattern] * 3, axis=-1)
                face_roi = np.clip(face_roi.astype(np.float32) + pattern, 0, 255).astype(np.uint8)

            # Pattern 3: Random pixel blocks
            for _ in range(20):
                block_size = np.random.randint(2, 8)
                bx = np.random.randint(0, w - block_size)
                by = np.random.randint(0, h - block_size)
                face_roi[by:by + block_size, bx:bx + block_size] = np.random.randint(0, 255,
                                                                                     (block_size, block_size, 3))

            frame[y:y + h, x:x + w] = face_roi
        return frame

    def color_chaos(self, frame, faces):
        """Extreme color manipulation that breaks color-based recognition"""
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            # Convert to different color spaces and manipulate
            # HSV manipulation
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV).astype(np.float32)

            # Random hue shift (major color change)
            hue_shift = np.random.randint(0, 180)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

            # Extreme saturation changes
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.3, 2.0), 0, 255)

            # Value channel noise
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.randint(-40, 40, hsv.shape[:2]), 0, 255)

            face_roi = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # LAB manipulation
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            lab = lab.astype(np.float32)

            # Mess with lightness channel
            lab[:, :, 0] = np.clip(lab[:, :, 0] * np.random.uniform(0.7, 1.3), 0, 255)

            # Add noise to color channels
            lab[:, :, 1] += np.random.randint(-30, 30, lab.shape[:2])
            lab[:, :, 2] += np.random.randint(-30, 30, lab.shape[:2])
            lab = np.clip(lab, 0, 255)

            face_roi = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # YCrCb manipulation
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            ycrcb = ycrcb.astype(np.float32)

            # Mess with luminance
            ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] * np.random.uniform(0.6, 1.4), 0, 255)

            # Chrominance noise
            ycrcb[:, :, 1] += np.random.randint(-25, 25, ycrcb.shape[:2])
            ycrcb[:, :, 2] += np.random.randint(-25, 25, ycrcb.shape[:2])
            ycrcb = np.clip(ycrcb, 0, 255)

            face_roi = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

            frame[y:y + h, x:x + w] = face_roi
        return frame

    def triple_threat(self, frame, faces):
        """Combine the three strongest techniques"""
        frame = self.neural_scrambler(frame, faces)
        frame = self.pixel_destroyer(frame, faces)
        frame = self.color_chaos(frame, faces)
        return frame


class CameraManager:
    def __init__(self):
        self.cap = None

    def initialize_camera(self):
        """Initialize camera"""
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Try different camera indices
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        return True
                cap.release()
            except:
                continue
        return False

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


# === STREAMLIT APP ===
def main():
    st.set_page_config(page_title="Strong Face Evasion", layout="wide")
    st.title("💥 Strong Face Recognition Evasion")
    st.markdown("### Aggressive techniques that will definitely break face recognition!")

    # Initialize systems
    if 'systems_initialized' not in st.session_state:
        st.session_state.face_recog = FaceRecognitionSystem()
        st.session_state.face_evader = StrongFaceEvader()
        st.session_state.evasion_tech = StrongEvasionTechniques()
        st.session_state.camera = CameraManager()
        st.session_state.recog_running = False
        st.session_state.evasion_running = False
        st.session_state.current_technique = "triple_threat"
        st.session_state.detected_people = set()
        st.session_state.systems_initialized = True

    # Sidebar configuration
    st.sidebar.header("💀 Strong Evasion Settings")

    # Dataset path input
    dataset_path = st.sidebar.text_input(
        "Dataset Path (optional)",
        value="C:/Users/Riyansh Saxena/Downloads/Face recognition system",
        help="Path to folder containing person folders with face images"
    )

    if st.sidebar.button("🔄 Load Training Data"):
        with st.spinner("Training face recognition model..."):
            success = st.session_state.face_recog.load_training_data(dataset_path)
            if success:
                st.sidebar.success("Model trained successfully!")
            else:
                st.sidebar.warning("Using default recognizer")

    # Evasion technique selection
    technique = st.sidebar.selectbox(
        "💥 Choose STRONG Evasion Technique",
        options=list(st.session_state.evasion_tech.techniques.keys()),
        format_func=lambda x: x.replace('_', ' ').title() + " 💀",
        help="Select which strong evasion technique to apply"
    )
    st.session_state.current_technique = technique

    st.sidebar.warning(f"**{st.session_state.evasion_tech.techniques[technique]}**")

    # Main interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Live Camera Feed")
        video_placeholder = st.empty()

        # Control buttons
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🎯 Start Recognition", use_container_width=True, type="primary"):
                st.session_state.recog_running = True
                st.session_state.evasion_running = False
        with col1b:
            if st.button("💀 Start EVASION", use_container_width=True, type="secondary"):
                if st.session_state.recog_running:
                    st.session_state.evasion_running = True
                else:
                    st.warning("Start recognition first!")
        with col1c:
            if st.button("⏹️ Stop All", use_container_width=True):
                st.session_state.recog_running = False
                st.session_state.evasion_running = False

    with col2:
        st.subheader("🛡️ System Status")

        if st.session_state.recog_running:
            if st.session_state.evasion_running:
                st.error(f"""
                **💀 EVASION ACTIVE - RECOGNITION SHOULD FAIL**

                **Technique:** {technique.replace('_', ' ').title()}
                **Strength:** Maximum aggression
                **Expected:** All faces should become 'Unknown'
                """)
            else:
                st.success("""
                **🟢 RECOGNITION ACTIVE**

                **Status:** Identifying faces normally
                **Evasion:** Ready to destroy recognition
                """)
        else:
            st.info("""
            **🔵 SYSTEM READY**

            Start recognition to begin monitoring
            Then activate STRONG evasion to break recognition
            """)

        # Recognition results
        st.markdown("---")
        st.subheader("👥 Recognition Results")
        results_placeholder = st.empty()

    # Camera initialization
    if not st.session_state.camera.initialize_camera():
        st.error("❌ Camera not found. Please check your camera connection.")
        return

    # Processing loop
    if st.session_state.recog_running:
        while st.session_state.recog_running:
            ret, frame = st.session_state.camera.get_frame()
            if not ret:
                st.error("Camera error")
                break

            frame = cv2.flip(frame, 1)

            # Apply evasion if active
            if st.session_state.evasion_running:
                # Detect faces for evasion
                evasion_faces = st.session_state.face_evader.detect_faces(frame)
                if evasion_faces:
                    technique_func = getattr(st.session_state.evasion_tech, st.session_state.current_technique)
                    frame = technique_func(frame, evasion_faces)

                    # Mark evaded faces with warning symbols
                    for (x, y, w, h) in evasion_faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(frame, "💀 EVADED", (x, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Perform face recognition on processed frame
            recognition_results = st.session_state.face_recog.recognize_faces(frame)

            current_detections = set()
            recognized_count = 0

            # Draw recognition results
            for result in recognition_results:
                x, y, w, h = result['coords']
                name = result['name']
                color = result['color']
                status = result['status']

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if status == "recognized":
                    current_detections.add(name)
                    recognized_count += 1

            # Update detected people set
            st.session_state.detected_people.update(current_detections)

            # Add status overlay
            status_text = "RECOGNITION ACTIVE"
            if st.session_state.evasion_running:
                status_text += " + STRONG EVASION ACTIVE 💀"
                status_color = (0, 0, 255)  # Red for evasion active
            else:
                status_color = (255, 255, 255)  # White for normal

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Faces: {len(recognition_results)} | Recognized: {recognized_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Update results sidebar
            with results_placeholder.container():
                if st.session_state.detected_people:
                    st.success(f"✅ Recognized {len(st.session_state.detected_people)} people:")
                    for name in sorted(st.session_state.detected_people):
                        st.markdown(f"**• {name}**")
                else:
                    st.info("👀 No recognized people yet")

                st.metric("Current Faces", len(recognition_results))
                st.metric("Recognized Now", recognized_count)

                if st.session_state.evasion_running:
                    if recognized_count > 0:
                        st.error(f"❌ Evasion failing! {recognized_count} faces still recognized")
                        st.warning("Try a different evasion technique or increase strength")
                    else:
                        st.success("🎉 Perfect! Evasion is working - no faces recognized!")
                else:
                    st.info("Recognition running normally")

            time.sleep(0.03)

    else:
        video_placeholder.info("Click 'Start Recognition' to begin")

        # Show final summary if we had a session
        if hasattr(st.session_state, 'detected_people') and st.session_state.detected_people:
            with results_placeholder.container():
                st.subheader("📊 Previous Session Summary")
                st.success(f"Recognized {len(st.session_state.detected_people)} people:")
                for name in sorted(st.session_state.detected_people):
                    st.markdown(f"• {name}")
                st.session_state.detected_people.clear()

    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.header("💪 Strong Evasion Techniques")
    st.sidebar.markdown("""
    **💀 Neural Scrambler:** 
    - Strong noise + color separation
    - Local histogram destruction

    **🎯 Pixel Destroyer:**
    - Extreme pixel-level attacks
    - Block scrambling + edge destruction

    **🎭 Feature Destroyer:**
    - Targets eyes, nose, mouth
    - Aggressive region attacks

    **⚡ Adversarial Attack:**
    - Pattern-based noise
    - Breaks neural networks

    **🌈 Color Chaos:**
    - Extreme color manipulation
    - Multi-color space attacks

    **💥 Triple Threat:**
    - Combines 3 strongest methods
    - Maximum destruction
    """)


if __name__ == "__main__":
    main()