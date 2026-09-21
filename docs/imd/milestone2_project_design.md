# CricLens

**Team Members**

| Name | Enrollment No. |
|---|---|
| Shikhar Srivastava | S24CSEU0962 |
| Vanshika Gupta | S24CSEU1135 |

---

# PROJECT DESIGN

**Project Title:** Cricket Batting Shot Classification and Technique Scoring using Deep Learning

Coaching feedback on batting technique is normally given by eye, by someone experienced enough to
notice a dropped shoulder or a late foot. That expertise is scarce, and video alone does not
substitute for it. CricLens takes a single batting clip and returns three things: the shot that was
played, a score for how well each part of the body contributed to it, and feedback written in
language a batter can act on. The system works from body movement rather than from how the frame
looks, so it generalises across grounds, kits and broadcast styles.

## 1. Objectives & Methodology of Proposed Work

**Objectives**

- Recognise which shot a batter has played, from the motion of their body through the stroke.
- Find the striker automatically in broadcast footage, where fielders, umpires and the wicketkeeper
  share the frame with them.
- Represent the stroke as a sequence of body joints spanning backlift, contact and follow-through.
- Compare LSTM, GRU and Transformer models on that sequence, to establish which architecture
  captures the temporal structure of a shot best.
- Score technique separately for the head, shoulders, hands, hips and feet, rather than returning a
  single overall verdict.
- Turn those scores into specific written feedback, delivered through an interface where a user
  uploads a clip and gets a reply.

**Methodology**

1. **Dataset collection.** Gather batting clips from public cricket datasets and standardise them to
   one resolution, encoding and frame format.
2. **Deduplication and splitting.** Remove repeated broadcast clips using perceptual hashing, then
   split by match, so that no match contributes to more than one split.
3. **Label unification.** Reconcile the different label sets used by each source into one shared
   taxonomy of shot classes.
4. **Object detection.** Train a YOLO detector to locate the ball, bat and stumps, which supply the
   geometric cues the later stages depend on.
5. **Batter identification.** Detect and track everyone in the clip, then train a classifier to
   decide which of those tracks is the striker.
6. **Frame preprocessing.** Apply contrast enhancement, noise filtering and edge padding so the
   batter's crop is fit for pose estimation.
7. **Pose estimation.** Run a Vision Transformer pose model over the crop to extract body joints for
   every frame in the stroke window.
8. **Sequence normalisation.** Resample the joint tracks to a common frame rate and normalise them
   for batter size and position, so the model learns posture instead of camera framing.
9. **Shot classification.** Train LSTM, GRU and Transformer models on the normalised sequences and
   compare them under identical conditions.
10. **Technique scoring.** Train a variational autoencoder on the same sequences, with a regression
    head predicting a score for each body part.
11. **Evaluation.** Judge classification by accuracy, precision, recall, F1-score and the confusion
    matrix, and scoring by error against expert annotations, with pose overlays checked by eye on
    sampled clips.
12. **Application.** Serve the trained pipeline behind an interface that returns the shot, the scores
    and the written feedback.

## 2. Relevance of Algorithms/Techniques

1. **Perceptual hashing and match-grouped splitting.** Broadcast clips of the same delivery recur
   across sources and share camera angle, lighting and batter; separating them is what makes the
   reported accuracy mean anything.
2. **YOLO detection.** Fast enough to run on every frame of every clip, which a two-stage detector
   would not be at this scale.
3. **Object tracking.** Holds a consistent identity for each person, so the batter is chosen once per
   clip instead of being re-guessed frame by frame.
4. **Gradient boosting.** Learns the striker from a handful of track features and handles the cases
   where fixed positional rules fail, such as a non-striker closer to the camera.
5. **CLAHE.** Lifts local contrast around the bat and body edges without amplifying noise across the
   whole frame, which global equalisation would.
6. **Vision Transformer pose estimation.** Self-attention relates distant body parts to one another,
   keeping joints stable through the occlusion and motion blur normal in broadcast footage.
7. **LSTM and GRU.** A shot is defined by the order of movements, not by any one frame; gating lets
   these models carry that information across the full stroke without vanishing gradients.
8. **Transformer self-attention.** Links the backlift straight to the follow-through however far
   apart they fall, and reads the sequence in parallel rather than step by step.
9. **Variational autoencoder.** Learns a continuous latent space of batting posture, so technique can
   be measured as distance from sound movement rather than sorted into good and bad.
10. **Cross-entropy, Adam and the evaluation metrics.** Cross-entropy suits multi-class
    classification, Adam adapts the step size during training, and the confusion matrix shows which
    shots are being mistaken for each other, which a single accuracy figure hides.

**Project Flow:** Video clip → Preprocessing → Object detection → Batter identification → Pose
estimation → Sequence normalisation → Shot classification (LSTM / GRU / Transformer) → Technique
scoring (VAE) → Coaching feedback → Interface
