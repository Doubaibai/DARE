# Deception Detection in Videos

We present a system for covert automated deception detection using information available in a video. We study the importance of different modalities like vision, audio and text for this task. On the vision side, our system uses classifiers trained on low level video features which predict human micro-expressions. We show that predictions of high-level micro-expressions can be used as features for deception prediction. Surprisingly, IDT (Improved Dense Trajectory) features which have been widely used for action recognition, are also very good at predicting deception in videos. We fuse the score of classifiers trained on IDT features and high-level micro-expressions to improve performance. MFCC (Mel-frequency Cepstral Coefficients) features from the audio domain also provide a significant boost in performance, while information from transcripts is not very beneficial for our system. Using various classifiers, our automated system obtains an AUC of 0.877 (10-fold cross-validation) when evaluated on subjects which were not part of the training set. 
Even though state-of-the-art methods use human annotations of micro-expressions for deception detection, our fully automated approach outperforms them by 5%. When combined with human annotations of micro-expressions, our AUC improves to 0.922. We also present results of a user-study to analyze how well do average humans perform on this task, what modalities they use for deception detection and how they perform if only one modality is accessible.

## Demo
Try our demo! You can select video, click upload to play the video. Also check predictions under different modalities!
[![Imgur](https://i.imgur.com/UjOZCP8.png)](http://www.cs.dartmouth.edu/~mbolonkin/dare/demo/)

## Framework

![Image of Framework](https://Doubaibai.github.io/Framework2.png)

## Experimental Results

We evaluate our automated deception detection approach on a [real-life deception detection database (Perez-Rosas et al. 2015)](http://web.eecs.umich.edu/~zmohamed/PDFs/Trial.ICMI.pdf). 

### Deception Detection Results
- Fully Automated Approach
![Image of table1](https://i.imgur.com/j5D2uf6.png)
- With Ground Truth Micro-Expression Features
![Image of table2](https://i.imgur.com/fYryR8M.png)

### User Study
To test human performance on this task, we perform user studies using AMT (Amazon Mechanical Turk). Turkers are provided each modality (i.e. image, audio, transcripts) without other modalities. We also ask other turkers to make decision with the whole video with all modalities and ask them which modality contributes most to their decision. 

- The importance of modalities (Turkers think) in making decisions.
<img src="https://i.imgur.com/IkoAlbJ.png" width="480">

- Human performance in deception detection using different modalities is compared with our automated system and our system with Ground Truth micro-expressions.
<img src="https://i.imgur.com/K5mTtpu.png" width="480">
