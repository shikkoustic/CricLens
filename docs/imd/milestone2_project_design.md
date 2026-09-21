# CricLens

**Team Members**
- Shikhar Srivastava
- *(teammate name)*

---

# PROJECT DESIGN

**Project Title:** Cricket Batting Shot Classification and Technique Scoring using Deep Learning

## 1. Objectives & Methodology of Proposed Work

- To develop a deep learning system that identifies the cricket shot played in a batting video and scores the batter's technique.
- To automatically locate the striking batter in a broadcast frame without manual annotation.
- To extract the batter's body pose as a sequence of joints across the stroke.
- To use LSTM, GRU and Transformer models to learn the temporal movement pattern that defines each shot.
- To score technique for individual body parts using a variational autoencoder.
- To provide an interface where a user uploads a clip and receives the shot, the scores and coaching feedback.

**Methodology**

1. **Dataset Collection:** Collect cricket batting clips from public datasets and standardise them to a common resolution and format.
2. **Duplicate Removal and Splitting:** Remove repeated broadcast clips using perceptual hashing, and split the data by match so that no match appears in more than one split.
3. **Label Unification:** Map each source's own labels onto a shared set of shot classes.
4. **Object Detection:** Train a YOLO detector to locate the ball, bat and stumps.
5. **Batter Identification:** Detect and track every person in the clip, and train a classifier to select the striker.
6. **Frame Preprocessing:** Apply contrast enhancement, noise filtering and edge padding to prepare the batter crop for pose estimation.
7. **Pose Estimation:** Use a Vision Transformer pose model to extract body joints per frame across the stroke window.
8. **Sequence Normalisation:** Resample joint sequences to a common frame rate and normalise them for batter size and position.
9. **Shot Classification:** Train LSTM, GRU and Transformer models on the joint sequences to classify the shot.
10. **Technique Scoring:** Train a variational autoencoder on the pose sequences, with a regression head predicting scores for individual body parts.
11. **Evaluation:** Measure classification using accuracy, precision, recall, F1-score and a confusion matrix, and scoring using error against expert annotations.
12. **Application:** Integrate the trained models into an interface that returns the shot, the scores and written coaching feedback.

## 2. Relevance of Algorithms/Techniques

1. **Perceptual Hashing and Match-Grouped Splitting:** Ensures the same delivery does not appear in both training and test data, so the reported accuracy is meaningful.
2. **YOLO Object Detection:** Locates the ball, bat and stumps quickly enough to run across every frame of every clip.
3. **Object Tracking:** Maintains a consistent identity for each person, so the batter is chosen once per clip rather than per frame.
4. **Gradient Boosting:** Selects the striker from a small set of track features more reliably than fixed positional rules.
5. **CLAHE:** Enhances local contrast so the bat and body edges stay visible in poor broadcast footage, without amplifying noise across the whole frame.
6. **Vision Transformer Pose Estimation:** Self-attention relates distant body parts to each other, keeping joint estimates stable under occlusion and motion blur.
7. **LSTM and GRU:** A shot is defined by the order of movements, not by a single frame; gated recurrent models carry this information across the full stroke without vanishing gradients.
8. **Transformer Self-Attention:** Relates the backlift directly to the follow-through regardless of the time between them, and processes the sequence in parallel.
9. **Variational Autoencoder:** Learns a latent space of batting posture, so deviations from sound technique can be measured rather than only classified.
10. **Cross-Entropy, Adam and Evaluation Metrics:** Cross-entropy suits multi-class shot classification, Adam updates weights efficiently during training, and accuracy, F1-score and the confusion matrix show which shots are being confused.

**Project Flow:** Video Clip → Preprocessing → Object Detection → Batter Identification → Pose Estimation → Sequence Normalisation → Shot Classification (LSTM / GRU / Transformer) → Technique Scoring (VAE) → Coaching Feedback → Interface
