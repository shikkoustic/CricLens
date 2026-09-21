# CricLens

**Team Members**
- Shikhar Srivastava
- *(teammate name)*

---

# PROJECT DESIGN

**Project Title:** CricLens — Deep Learning Based Batting Technique Analysis and Coaching Feedback from Cricket Video

## 1. Objectives & Methodology of Proposed Work

**Objectives**

- To develop a deep learning system that takes a cricket batting video clip and returns the shot played, a per-body-part technique score, and readable coaching feedback.
- To automatically identify the striking batter among all the people visible in a broadcast frame, without any manual annotation at inference time.
- To extract the batter's body pose as a temporal sequence of joints spanning the stroke, from just before ball release to follow-through.
- To classify the stroke into eight shot classes using sequence models, and to compare RNN, LSTM, GRU and Transformer architectures on the same data.
- To score batting technique separately for head, shoulders, hands, hips and feet using a variational autoencoder trained on expert-annotated scores.
- To audit the trained models for handedness bias between left-handed and right-handed batters.
- To convert the predicted scores into natural-language coaching feedback and serve the whole pipeline through a web interface.

**Methodology**

1. **Dataset Collection:** Assemble batting clips from six public cricket-video datasets. 24,032 decoded clips are reduced to 22,420 unique clips after duplicate removal, and every clip is standardised to a common resolution and encoding.
2. **Duplicate Removal:** Detect near-identical broadcast clips using perceptual hashing with locality-sensitive hashing for recall, followed by a stricter time-aligned hash-sequence check for verification.
3. **Label Unification and Splitting:** Map each source's own label set onto a shared taxonomy of eight shot classes plus shot side. Split the data by match, so that no match and no duplicate cluster appears in more than one split, preventing leakage.
4. **Object Detection:** Train a YOLO-family detector on 23,799 annotated frames to locate the ball, bat and stumps, giving the geometric cues the later stages depend on.
5. **Batter Identification:** Detect and track every person in the clip, compute features describing each track relative to the others in the same clip, and train a gradient-boosting classifier to pick the striker.
6. **Pose Estimation:** Run a Vision-Transformer-based pose model on the selected batter to extract 17 body joints per frame, over a stroke window running from 0.8 s before bat-ball contact to 0.6 s after, truncated at camera shot changes.
7. **Input Conditioning:** Apply contrast-limited adaptive histogram equalisation to the luminance channel before bat detection, and zero-padding rather than clamping when the batter's crop extends past the frame edge, so that crop geometry is preserved and no false body evidence is introduced.
8. **Sequence Normalisation:** Resample joint trajectories to a single frame rate, and normalise joint coordinates by batter height and position so the model learns posture rather than camera framing.
9. **Shot Classification:** Train and compare RNN, LSTM, GRU and Transformer sequence models over the normalised joint sequences, using softmax output with cross-entropy loss.
10. **Technique Scoring:** Train a variational autoencoder over the pose sequences to learn a latent representation of batting posture, with a regression head predicting expert 1–10 scores for head, shoulders, hands, hips and feet.
11. **Bias Audit:** Evaluate both models separately on left-handed and right-handed batters and report the gap.
12. **Metric Calibration:** Recover real-world scale from pitch and crease geometry so that stride length and bat-swing speed can be reported in physical units rather than pixels.
13. **Coaching Feedback:** Pass the predicted per-part scores and detected faults to a language model that produces short, specific coaching advice.
14. **Evaluation:** Measure shot classification using accuracy, precision, recall, F1-score and a confusion matrix; measure scoring using mean absolute error and correlation against expert scores; and verify all model outputs by rendering pose overlays on sampled clips rather than trusting metrics alone.
15. **Application:** Deploy the trained pipeline behind a web interface where a user uploads a clip and receives the shot, the scores and the written feedback.

## 2. Relevance of Algorithms/Techniques

1. **Perceptual Hashing with LSH:** Broadcast datasets repeat the same delivery across sources; hashing removes those duplicates so the same stroke cannot appear in both training and test data.
2. **Match-Grouped Splitting:** Deliveries from one match share camera angle, lighting and batter, so splitting by match is what makes the reported accuracy meaningful.
3. **YOLO Object Detection:** A single-stage detector locates the ball, bat and stumps fast enough to run over every frame of every clip, achieving 0.81 mAP50 on unseen test data.
4. **Multi-Object Tracking:** Maintains a persistent identity for each person across frames, which is what allows a batter to be chosen once per clip rather than per frame.
5. **Gradient Boosting:** Chooses the striker from a small set of tabular track features, reaching 91.2% accuracy on unseen matches where hand-written positional rules reach only 57%.
6. **Vision Transformer Pose Estimation:** Self-attention relates distant body parts to each other, which keeps joint estimates stable through the occlusion and motion blur typical of broadcast footage.
7. **Zero Padding at Frame Edges:** Preserves the aspect ratio of the batter crop, and the padded region carries no body-like texture that could be mistaken for a limb; this reduces joint error on edge cases by 20–60%.
8. **CLAHE (Adaptive Histogram Equalisation):** Recovers local contrast on the bat without amplifying noise globally, raising true bat detections inside the batter box from 11% to 19% and 7% to 22% in controlled tests.
9. **Adaptive Median Filtering:** Removes impulse noise from user-uploaded footage while preserving the edges the pose model relies on, outperforming a plain median filter on this task.
10. **RNN, LSTM and GRU:** A cricket shot is defined by the order of movements, not by any single frame; recurrent models learn these temporal dependencies, and the gated variants carry information across the full stroke without the vanishing-gradient problem of a plain RNN.
11. **Transformer Self-Attention:** Relates the backlift directly to the follow-through regardless of the gap between them, and processes the sequence in parallel rather than step by step.
12. **Variational Autoencoder:** Learns a smooth latent space of batting posture, so technique can be scored by position within that space and deviations from sound technique become measurable rather than merely classified.
13. **Hough Transform and Edge Detection:** Recovers the crease lines whose real-world dimensions are fixed and known, which is what converts pixel measurements into centimetres and metres per second.
14. **Cross-Entropy, Adam and Evaluation Metrics:** Cross-entropy suits multi-class shot classification, Adam adapts the learning rate during training, and accuracy, F1-score and the confusion matrix expose which shot classes are being confused rather than reporting a single number.

**Project Flow:** Video Clip → Standardisation & Deduplication → Object Detection → Person Tracking → Batter Identification → Pose Estimation → Sequence Normalisation → Shot Classification (RNN / LSTM / GRU / Transformer) → Technique Scoring (VAE) → Coaching Feedback → Web Interface
