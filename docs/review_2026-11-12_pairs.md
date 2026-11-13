# `_Thesis_12.11.pdf`: every changed passage

Before / after in the LaTeX source (Overleaf `332e57b`), by chapter. The comments behind them are in `docs/review_2026-11-12.md`. An empty *after* means the passage was removed or moved.

## Chapter 1: Introduction (4 changes)

1. **Before:** To the author's knowledge,  
   **After:** To my knowledge,

2. **Before:** This thesis addresses it with the following question:  
   **After:** I address it with the following question:

3. **Before:** Four research questions (RQ1 to RQ4) structure the work. Chapter~\ref{ch:discussion} answers each of them (Section~\ref{sec:disc_rq}). \begin{description} \item[RQ1] How well do the emotions expressed by a soundtrack predict its film's genre? \item[RQ2] Which emotions characterise each genre, and how far do these emotional signatures hold on a second, independent dataset? \item[RQ3] How does a pipeline that predicts genre from emotions compare with classifying genre directly from the audio representation the emotions are computed from? \item[RQ4] How well do the two pipelines generalise when a model trained on one dataset is applied to another? \end{description} RQ1 asks how strong the signal is. It is answered first with emotions rated by human listeners and then with emotions predicted from audio. RQ2 concerns the form of the signal: which emotions distinguish each genre, and whether the same emotional signatures appear in a second collection of films. This examines whether the emotional step can be interpreted in the way the motivation above suggests. RQ3 is the central comparison of the thesis, made within one dataset. RQ4 asks how the two pipelines perform when a model trained on one collection of films is applied to another without access to any of its labels. RQ2 and RQ4 both use the second dataset, but they address different questions. RQ2 asks whether the link between emotions and genres stays the same there. RQ4 asks whether a genre classifier still works there.  
   **After:** (research questions as prose)

4. **Before:** Each film is described by audio features of the music cues it uses, but there are no emotion ratings and no audio.  
   **After:** Each film is described by features of the music cues it uses, published by the authors: VGGish embeddings, a learned audio representation, and hand-crafted music information retrieval (MIR) descriptors. The dataset contains no emotion ratings and no audio.

## Chapter 2: Fundamentals (16 changes)

5. **Before:** It thereby retains roughly the information that the ear retains, which is one reason for its wide use.  
   **After:** It thereby retains roughly the information that the ear retains, which is one reason for its wide use. Figure~\ref{fig:spectrogram} shows both forms for one clip of the dataset used here. \begin{figure}[!htb] \centering \includegraphics[width=\textwidth]{figures/spectrogram.pdf} \caption{Waveform and log-mel spectrogram of the first ten seconds of a film-music clip.} \label{fig:spectrogram} \end{figure} The clip in Figure~\ref{fig:spectrogram} comes from the score of the Horror film \emph{Hellraiser}, and its listeners rated it among the most fearful of the Horror clips. The waveform at the top holds 160{,}000 values. The spectrogram below it holds one value per mel band and 10\,ms step, with brighter colours for more energy. Its settings are those of the VGGish front end: 64 mel bands between 125\,Hz and 7.5\,kHz, and windows of 25\,ms.

6. **Before:** The thesis uses five such networks:  
   **After:** I use five such networks:

7. **Before:** This thesis instead treats emotion as a possible reason for that link.  
   **After:** I instead treat emotion as a possible reason for that link.

8. **Before:** This thesis hypothesises that this bottleneck retains  
   **After:** I hypothesise that this bottleneck retains

9. **Before:** It further hypothesises that it does so  
   **After:** I further hypothesise that it does so

10. **Before:** Every split in this thesis is therefore \emph{grouped by film}: all clips of a film stay on the same side. Section~\ref{sec:evaluation_protocol} reports how large this effect is on the data used here.  
   **After:** I therefore group every split \emph{by film}: all clips of a film stay on the same side. Figure~\ref{fig:grouped_cv} illustrates this for five folds, and Section~\ref{sec:evaluation_protocol} reports how large the effect of grouping is on the data used here. \begin{figure}[!htb] \centering \begin{tikzpicture}[x=1.1cm, y=0.62cm, font=\footnotesize] \foreach \film in {1,...,10} {\node at (\film-0.5,0.05) {\scriptsize Film \film};} \foreach \row in {1,...,5} { \node[anchor=east] at (-0.15,-\row) {Fold \row}; \foreach \pos/\nclips in {0/3,1/5,2/2,3/4,4/6,5/3,6/2,7/5,8/4,9/3} { \pgfmathtruncatemacro{\fold}{int(\pos/2)+1} \ifnum\fold=\row \draw[fill=gray!45] (\pos+0.06,-\row-0.32) rectangle (\pos+0.94,-\row+0.32); \else \draw (\pos+0.06,-\row-0.32) rectangle (\pos+0.94,-\row+0.32); \fi \foreach \clip in {1,...,\nclips} { \pgfmathsetmacro{\xx}{\pos+0.06+0.88*\clip/(\nclips+1)} \draw[line width=0.7pt] (\xx,-\row-0.17) -- (\xx,-\row+0.17); } } } \draw[fill=gray!45] (0.06,-6.1) rectangle (0.5,-5.75); \node[anchor=west] at (0.55,-5.92) {test films}; \draw (2.6,-6.1) rectangle (3.04,-5.75); \node[anchor=west] at (3.09,-5.92) {training films}; \draw[line width=0.7pt] (5.4,-6.1) -- (5.4,-5.75); \node[anchor=west] at (5.45,-5.92) {one clip}; \end{tikzpicture} \caption{Five-fold cross-validation grouped by film: all clips of a film fall into the same fold.} \label{fig:grouped_cv} \end{figure} In Figure~\ref{fig:grouped_cv}, each row is one split of ten films into training and test data, and each mark inside a film stands for one of its clips. The two grey films of a row form the test fold. A film is never divided, so no clip of a test film is seen in training. The films differ in their number of clips, as they do in the real data.

11. **Before:** The coefficient of determination is the share of the variance in the human ratings that the model's predictions explain. $R^2 = 0$ corresponds to always predicting the mean rating, and $R^2 = 1$ would be perfect.  
   **After:** The coefficient of determination is the share of the variance in the human ratings that the model's predictions explain: \begin{equation} R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}, \end{equation} where \smallskip \noindent\begin{tabular}{@{}p{0.17\linewidth}p{0.79\linewidth}@{}} $n$ & the number of clips \\ $y_i$ & the rating of clip $i$, that is, the mean of its listeners \\ $\hat{y}_i$ & the rating the model predicts for clip $i$ \\ $\bar{y}$ & the mean rating over all clips \\ \end{tabular} \smallskip \noindent $R^2 = 0$ corresponds to always predicting the mean rating, and $R^2 = 1$ would be perfect.

12. **Before:** the agreement between the two groups, for example the intraclass correlation (ICC), estimates the share  
   **After:** the agreement between the two groups estimates the share

13. **Before:** No model can be expected to predict more than that share, so the reachable $R^2$ lies below 1.0. Section~\ref{subsec:rating_reliability} gives this ceiling for the ratings used here.  
   **After:** No model can be expected to predict more than that share, so the reachable $R^2$ lies below 1.0. Agreement between groups of raters is usually measured by the \emph{intraclass correlation} (ICC). Its \emph{consistency} form, ICC(C,1), asks whether two groups rank and space the items in the same way, and it ignores a constant offset between them. Its \emph{absolute-agreement} form, ICC(A,1), also counts such an offset as disagreement. Section~\ref{subsec:rating_reliability} gives both values and the resulting ceiling for the ratings used here.

14. **Before:** For a single genre, \emph{precision} is the share of items given that genre that really have it. \emph{Recall} is the share of items that really have it that were given it. $F_1$ is their harmonic mean. \emph{Macro} averaging computes $F_1$ per genre and averages these values with equal weight, so a rare genre counts as much as a common one.  
   **After:** For one genre, each decision of a classifier falls into one of four cases (Figure~\ref{fig:confusion}). A \emph{true positive} (TP) has the genre and is assigned it. A \emph{false positive} (FP) is assigned the genre without having it. A \emph{false negative} (FN) has the genre but is not assigned it, and a \emph{true negative} (TN) neither has it nor is assigned it. \begin{figure}[!htb] \centering \begin{tikzpicture}[font=\footnotesize, cell/.style={draw, minimum width=3.0cm, minimum height=1.05cm, align=center}] \node[cell, fill=gray!25] at (0,0) {true positive (TP)}; \node[cell] at (3.0,0) {false positive (FP)}; \node[cell] at (0,-1.05) {false negative (FN)}; \node[cell, fill=gray!25] at (3.0,-1.05) {true negative (TN)}; \node at (0,0.8) {has the genre}; \node at (3.0,0.8) {does not have it}; \node[anchor=east] at (-1.65,0) {assigned the genre}; \node[anchor=east] at (-1.65,-1.05) {not assigned}; \node[anchor=west] at (4.65,0) {$\rightarrow$ precision}; \node at (0,-1.95) {$\downarrow$ recall}; \end{tikzpicture} \caption{The four outcomes of a decision for one genre, and the two measures built from them.} \label{fig:confusion} \end{figure} In Figure~\ref{fig:confusion}, the columns give the truth and the rows the decision. Precision is read along the first row and recall down the first column: \begin{equation} \text{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}, \qquad \text{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \qquad F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}. \end{equation} Precision is the share of items given the genre that really have it. Recall is the share of items that really have the genre that were given it. $F_1$ is their harmonic mean, so it is high only when both are high. None of the three uses the true negatives. \emph{Macro} averaging computes $F_1$ for each genre and averages these values with equal weight: \begin{equation} \text{macro-}F_1 = \frac{1}{G} \sum_{g=1}^{G} F_{1,g}, \end{equation} where \smallskip \noindent\begin{tabular}{@{}p{0.17\linewidth}p{0.79\linewidth}@{}} TP, FP, FN & the numbers of true positives, false positives and false negatives of one genre \\ $G$ & the number of genres \\ $F_{1,g}$ & the $F_1$ score of genre $g$ \\ \end{tabular} \smallskip \noindent A rare genre therefore counts as much as a common one.

15. **Before:** Its $F_1$ for the genre is therefore about the share too, and its macro-$F_1$ is about the mean of the shares: (0.30 + 0.30 + 0.725 + 0.10 + 0.125) / 5 = 0.31.  
   **After:** Its $F_1$ for the genre is therefore about the share too, and its macro-$F_1$ is about the mean of the shares: \begin{equation} \text{macro-}F_1^{\,\text{random}} \approx \frac{1}{G} \sum_{g=1}^{G} p_g = \frac{0.30 + 0.30 + 0.725 + 0.10 + 0.125}{5} = 0.31, \end{equation} where $p_g$ is the share of films that carry genre $g$.

16. **Before:** For one genre, it is the probability that a randomly chosen film with the genre receives a higher predicted probability than a randomly chosen film without it.  
   **After:** For one genre, it is the probability that a randomly chosen film with the genre receives a higher predicted probability than a randomly chosen film without it: \begin{equation} \text{ROC-AUC} = \frac{1}{n_{+}\, n_{-}} \sum_{i=1}^{n_{+}} \sum_{j=1}^{n_{-}} \left[ \mathbbm{1}(s_i > s_j) + \tfrac{1}{2}\, \mathbbm{1}(s_i = s_j) \right], \end{equation} where \smallskip \noindent\begin{tabular}{@{}p{0.17\linewidth}p{0.79\linewidth}@{}} $n_{+}$, $n_{-}$ & the numbers of films with and without the genre \\ $s_i$, $s_j$ & the predicted probabilities of a film $i$ with the genre and a film $j$ without it \\ $\mathbbm{1}(\cdot)$ & 1 if the condition in brackets holds, and 0 otherwise \\ \end{tabular} \smallskip \noindent

17. **Before:** The all-positive classifier has a recall of 1 and a precision equal to the share $p$ of items with the genre, so its $F_1$ is $2p/(1+p)$.  
   **After:** The all-positive classifier has a recall of 1 and a precision equal to the share $p$ of items with the genre, so its $F_1$ is \begin{equation} F_1^{\,\text{all}} = \frac{2 \cdot p \cdot 1}{p + 1} = \frac{2p}{1+p}. \end{equation}

18. **Before:** Two tests are used to assess it, and they answer different questions.  
   **After:** Two tests are used to assess it, and they answer different questions. In this thesis the bootstrap is the main test for every comparison, and the corrected $t$-test is a stricter check beside it (Section~\ref{subsec:protocol_metrics}).

19. **Before:** The lead is probably real, but another sample of films could still eliminate it.  
   **After:** The lead may well be real, but another sample of films could still eliminate it.

20. **Before:** puts a lower limit on the $p$-value it can reach on a given dataset (Appendix~\ref{subsec:statistical_power}).  
   **After:** puts a lower limit on the $p$-value it can reach on a given dataset (Appendix~\ref{subsec:statistical_power}). The test needs one score per test fold. I therefore use it for the comparisons scored over clips, where every fold has such a score (Section~\ref{subsec:protocol_metrics}).

## Chapter 3: Related Work (9 changes)

21. **Before:** This is one reason why this thesis combines dimensional and categorical emotions  
   **After:** This is one reason why I combine dimensional and categorical emotions

22. **Before:** This thesis follows that suggestion and uses such encoders, frozen, to extract features.  
   **After:** I follow that suggestion and use such encoders, frozen, to extract features.

23. **Before:** This thesis can test that only indirectly,  
   **After:** I can test that only indirectly,

24. **Before:** This thesis groups its cross-validation by film to prevent this kind of leakage.  
   **After:** I group the cross-validation by film to prevent this kind of leakage.

25. **Before:** This thesis repeats their protocol on their own published features.  
   **After:** I repeat their protocol on their own published features.

26. **Before:** This thesis does not treat this result as evidence against using audio.  
   **After:** I do not treat this result as evidence against using audio.

27. **Before:** This thesis therefore groups all cross-validation by film, and it also quantifies the effect of this safeguard.  
   **After:** I therefore group all cross-validation by film, and I also measure the effect of this safeguard.

28. **Before:** This thesis takes up both, with its feature set that combines  
   **After:** I take up both, with a feature set that combines

29. **Before:** and with its analysis of which emotions mark which genres  
   **After:** and with an analysis of which emotions mark which genres

## Chapter 4: Methods (20 changes)

30. **Before:** The labels are very unbalanced: Drama applies to 239 clips, Documentary to 17.  
   **After:** The labels are very unbalanced: Drama applies to 239 clips, Documentary to 17. Figure~\ref{fig:data_overview} shows how the clips are spread over the films and the films over the genres. \begin{figure}[!htb] \centering \includegraphics[width=\textwidth]{figures/data_overview.pdf} \caption{Clips per film and films per genre in the eight-genre Eerola space.} \label{fig:data_overview} \end{figure} In Figure~\ref{fig:data_overview}, the left panel counts the films by their number of clips, and the right panel counts the films that carry each genre. A film contributes between 2 and 18 clips. Drama is carried by most films, whereas Comedy, Horror, Biography and Documentary rest on five films or fewer.

31. **Before:** (every $|d| \le 0.25$; Section~\ref{sec:eval_signatures})  
   **After:** (its strongest effect is $|d| = 0.25$; Section~\ref{sec:eval_signatures})

32. **Before:** \begin{align*}  
   **After:** \begin{align}

33. **Before:** \end{align*}  
   **After:** \end{align}

34. **Before:** Genre labels were added from the Internet Movie Database (IMDb), using  
   **After:** I added the genre labels from the Internet Movie Database (IMDb), using

35. **Before:** These six were taken from \citet{ma2021computational}, not chosen here. The rule they used to reduce the IMDb genres to them was recovered from their published list of films and genres, not assumed.  
   **After:** I took these six from \citet{ma2021computational} instead of choosing them. I recovered the rule they used to reduce the IMDb genres to them from their published list of films and genres instead of assuming it.

36. **Before:** because those eight were chosen as the ones frequent enough  
   **After:** because I chose those eight as the ones frequent enough

37. **Before:** The effect of this was tested. Each clip was split into consecutive 10.24\,s windows, and the AST embeddings were pooled over all of them  
   **After:** I tested the effect of this. I split each clip into consecutive 10.24\,s windows and pooled the AST embeddings over all of them

38. **Before:** All thirteen hidden states (the output of the convolutional encoder and the twelve transformer blocks) were tested on emotion regression  
   **After:** I tested all thirteen hidden states (the output of the convolutional encoder and the twelve transformer blocks) on emotion regression

39. **Before:** It was selected on the emotion task in an earlier run, where it was the best block, and then held fixed.  
   **After:** I selected it on the emotion task in an earlier run, where it was the best block, and then held it fixed.

40. **Before:** The regressor was chosen by experiment. A random forest, ridge regression and support vector regression (SVR) with a radial basis function (RBF) kernel were compared on all six representations,  
   **After:** I chose the regressor by experiment. I compared a random forest, ridge regression and support vector regression (SVR) with a radial basis function (RBF) kernel on all six representations,

41. **Before:** Second, the pipeline was fixed before the comparison of six representations and three regressors was run  
   **After:** Second, I fixed the pipeline before running the comparison of six representations and three regressors

42. **Before:** This training procedure was fixed before film-level scoring was adopted for the evaluation, and it was not re-tuned for it.  
   **After:** I fixed this training procedure before adopting film-level scoring for the evaluation, and I did not re-tune it for that.

43. **Before:** More flexible classifiers were also tried on the emotion features:  
   **After:** I also tried more flexible classifiers on the emotion features:

44. **Before:** The thesis therefore combines the two frameworks, and it reports that a reduction to fewer emotions would not have lowered accuracy.  
   **After:** I therefore combine the two frameworks, and I report that a reduction to fewer emotions would not have lowered accuracy.

45. **Before:** The thesis therefore keeps macro-$F_1$ as the headline, because it measures the decisions, and reads ROC-AUC beside it as the check.  
   **After:** I therefore keep macro-$F_1$ as the headline, because it measures the decisions, and read ROC-AUC beside it as the check.

46. **Before:** Beside the $p$-values, the thesis gives in how many of the ten repeats  
   **After:** Beside the $p$-values, I report in how many of the ten repeats

47. **Before:** Before any result was interpreted, the extraction pipeline was checked from end to end on an easy side task:  
   **After:** Before interpreting any result, I checked the extraction pipeline from end to end on an easy side task:

48. **Before:** Two questions about the main metric were checked on the five-genre protocol,  
   **After:** I checked two questions about the main metric on the five-genre protocol,

49. **Before:** so it was run in a separate environment  
   **After:** so I ran it in a separate environment

## Chapter 5: Evaluation (39 changes)

50. **Before:** In Figure~\ref{fig:stage1_r2}, the last column is the mean over the eight emotions.  
   **After:** In Figure~\ref{fig:stage1_r2}, the panel on the right gives the mean over the eight emotions.

51. **Before:** Two results stand out. The three general-purpose spectrogram embeddings, CLAP, VGGish and AST, do not differ substantially ($R^2$ between 0.545 and 0.558). MusiCNN, pretrained on music, follows closely (0.512). The 103 hand-crafted descriptors are weaker (0.476), and the waveform model wav2vec~2.0 recovers far less than any other (0.299). The emotions also differ in difficulty. For the three strongest representations, tension, energy and anger are the easiest ($R^2$ 0.608 to 0.661). Valence and fear follow (0.559 to 0.603). Sadness (0.428 to 0.438) and happiness (0.302 to 0.321) are the hardest.  
   **After:** Two results stand out. The three general-purpose spectrogram embeddings, CLAP, VGGish and AST, do not differ substantially, and MusiCNN, pretrained on music, follows closely. The hand-crafted descriptors are weaker, and the waveform model wav2vec~2.0 recovers far less than any other representation. The emotions also differ in difficulty. For the three strongest representations, tension, energy and anger are the easiest to predict, and sadness and happiness the hardest.

52. **Before:** intraclass correlation for consistency ICC(C,1) $=0.897$).  
   **After:** intraclass correlation for consistency ICC(C,1) $=0.897$; Section~\ref{sec:fund_evaluation}).

53. **Before:** \begin{figure}[t] \centering \includegraphics[width=\textwidth]{figures/indomain_eerola5.pdf} \caption{Genre classification on the five-genre Eerola subset: film-level macro-$F_1$ of every pipeline with 95\,\% intervals.} \label{fig:indomain_eerola5} \end{figure}  
   **After:** (removed or moved)

54. **Before:** \section{Stage 2: Genre from Emotions} \label{sec:eval_genre}  
   **After:** \section{Stage 2: Genre from Emotions} \label{sec:eval_genre} \begin{figure}[!htb] \centering \includegraphics[width=\textwidth]{figures/indomain_eerola5.pdf} \caption{Genre classification on the five-genre Eerola subset: film-level macro-$F_1$ of every pipeline with 95\,\% intervals.} \label{fig:indomain_eerola5} \end{figure}

55. **Before:** With the human ratings as input, the classifier reaches a film-level macro-$F_1$ of 0.491 [0.411, 0.552]. The full emotion pipeline reaches 0.460 [0.385, 0.514]. Both lie far above the most-frequent baseline (0.168) and above the base-rate random guess (0.309). Scored over clips, the same predictions reach 0.416 and 0.405.  
   **After:** With the human ratings as input, the classifier reaches a film-level macro-$F_1$ of 0.491, and the full emotion pipeline reaches 0.460. Both lie well above the two chance floors, and both are lower when the same predictions are scored over clips.

56. **Before:** Predicted emotions perform only slightly worse than the ratings. At film level the ratings score 0.032 higher (difference $-0.032$ [$-0.090$, $+0.021$]). They are ahead in all ten repeats, but the difference is not significant ($p=0.277$). Over clips the two are on par ($-0.010$, $p=0.340$).  
   **After:** Predicted emotions perform only slightly worse than the ratings. The ratings are ahead in all ten repeats, but the difference is not significant at either level (Table~\ref{tab:power_cmp}).

57. **Before:** Used directly, the six representations reach MusiCNN 0.396, CLAP 0.390, VGGish 0.367, the hand-crafted descriptors 0.365, AST 0.318 and wav2vec~2.0 0.261 (Figure~\ref{fig:indomain_eerola5}). The PCA-8 control reaches 0.390. The emotion pipeline, at 0.460, is ahead of all of them, by 0.065 to 0.196.  
   **After:** Used directly, all six representations score below the emotion pipeline, by 0.065 to 0.196, and so does the PCA-8 control (Figure~\ref{fig:indomain_eerola5}).

58. **Before:** First, the emotion pipeline is ahead of the VGGish embedding it is computed from by $+0.093$, in 9 of ten repeats, and ahead of every other direct representation as well.  
   **After:** First, the emotion pipeline is ahead of the VGGish embedding it is computed from, and of every other direct representation as well.

59. **Before:** but not against VGGish ($p=0.052$), CLAP ($p=0.156$) and MusiCNN ($p=0.241$).  
   **After:** but not against VGGish, CLAP ($p=0.156$) and MusiCNN.

60. **Before:** The emotion pipeline's lead over the control, $+0.070$, is not significant ($p=0.089$). Over clips the control is on par with VGGish, and the emotion pipeline is ahead of it ($+0.042$, $p=0.020$).  
   **After:** The emotion pipeline's lead over the control is not significant at film level. Over clips the control scores like VGGish, and the emotion pipeline is ahead of it in the film bootstrap ($p=0.020$), though not in the Nadeau--Bengio test.

61. **Before:** The 18 emotion pipelines span 0.416 to 0.481, the six direct ones 0.261 to 0.396. This holds even  
   **After:** This holds even

62. **Before:** On wav2vec~2.0 it has the higher $R^2$ (0.339 against 0.299) but the lower genre score.  
   **After:** On wav2vec~2.0 it has the higher $R^2$ but the lower genre score.

63. **Before:** The differences between emotion pipelines are small compared with their spread. The standard deviation over the repeats is 0.019 to 0.048 for the emotion pipelines and 0.026 to 0.045 for the direct ones. The 95\,\% film-bootstrap interval  
   **After:** The differences between emotion pipelines are small compared with their spread over the repeats. The 95\,\% film-bootstrap interval

64. **Before:** The thesis therefore keeps VGGish with the random forest,  
   **After:** I therefore keep VGGish with the random forest,

65. **Before:** For Drama the highest score comes from the hand-crafted MIR features used directly (0.82 against 0.76 for the emotion pipeline), and for Horror it is direct CLAP (0.43 against 0.23).  
   **After:** For Drama the highest score comes from the hand-crafted MIR features used directly, and for Horror from direct CLAP.

66. **Before:** VGGish used directly rarely predicts it (recall 0.12), whereas the emotion pipeline identifies most Comedy films (recall 0.78), at a low precision (0.22).  
   **After:** VGGish used directly rarely predicts it, whereas the emotion pipeline identifies most Comedy films, at a low precision.

67. **Before:** There the emotion pipeline predicts the genre for too many films (precision 0.15 at a recall of 0.50) and performs no better than VGGish (0.23 against 0.23).  
   **After:** There the emotion pipeline predicts the genre for too many films and performs no better than VGGish.

68. **Before:** Table~\ref{tab:emotion_genre_signatures} gives the strongest signature of each of the eight Eerola genres. A positive $d$ means that the emotion is raised in the genre.  
   **After:** Figure~\ref{fig:signatures} gives the signature of each of the eight Eerola genres, computed from the human ratings. \begin{figure}[!htb] \centering \includegraphics[width=0.9\textwidth]{figures/signatures.pdf} \caption{Emotional signatures (Cohen's $d$) of the eight Eerola genres, from the human ratings.} \label{fig:signatures} \end{figure} In Figure~\ref{fig:signatures}, red marks an emotion rated higher in the genre than in the other clips, and blue a lower one. A box marks the strongest emotion of each genre, and the number after a genre is its number of clips.

69. **Before:** \begin{table}[!htb] \centering \caption{Dominant emotional signature (Cohen's $d$) of each of the eight Eerola genres.} \label{tab:emotion_genre_signatures} \begin{tabular}{llr} \toprule Genre & Dominant emotion (direction) & Cohen's $d$ \\ \midrule Horror & fear (high) & $+0.75$ \\ Biography & happiness (high) & $+0.67$ \\ Comedy & fear (low) & $-0.61$ \\ Drama & valence (high) & $+0.57$ \\ Documentary & anger (low) & $-0.51$ \\ Action & happiness (low) & $-0.50$ \\ Crime & fear (high) & $+0.40$ \\ Adventure & none with $|d|>0.25$ & -- \\ \bottomrule \end{tabular} \end{table}  
   **After:** (removed or moved)

70. **Before:** It separates the genres built on threat (Horror, $d=+0.75$, the strongest link between a genre and an emotion in the dataset; Crime, $+0.40$) from the light-hearted ones (Comedy, $-0.61$). Valence forms a second axis. It separates the genres with a positive tone (Drama, $+0.57$; Comedy) from those with a negative one (Horror, $-0.63$; Action).  
   **After:** It separates the genres built on threat (Horror, with the strongest link between a genre and an emotion in the dataset, and Crime) from the light-hearted ones (Comedy). Valence forms a second axis. It separates the genres with a positive tone (Drama, Comedy) from those with a negative one (Horror, Action).

71. **Before:** Adventure has no signature: no emotion exceeds $|d|=0.25$.  
   **After:** Adventure has no signature: its strongest emotion, fear, reaches only $|d|=0.25$.

72. **Before:** The same effect sizes are then computed for the six shared genres.  
   **After:** The same effect sizes are then computed for the six shared genres, from the Eerola ratings and from the predicted Blockbuster emotions.

73. **Before:** \begin{figure}[!htb] \centering \includegraphics[width=\textwidth]{figures/signatures.pdf} \caption{Emotional signatures (Cohen's $d$) of the six shared genres on Eerola (left) and Blockbuster (right).} \label{fig:signatures} \end{figure} In Figure~\ref{fig:signatures}, red marks an emotion raised in the genre and blue a lowered one. The Eerola matrix comes from the human ratings, the Blockbuster matrix from the predicted emotions.  
   **After:** (removed or moved)

74. **Before:** line marks the base-rate random guess (0.295).  
   **After:** line marks the base-rate random guess.

75. **Before:** In-domain on Blockbuster, the emotion pipeline (0.616 $\pm$ 0.022) is on par with direct audio. It is nominally ahead of the film-average embedding (0.593; $+0.023$, Nadeau--Bengio $p=0.536$) and slightly behind the vote over cues (0.621). None of these differences is significant. The PCA-8 control (0.587) is on par with them as well.  
   **After:** In-domain on Blockbuster, the emotion pipeline performs about as well as direct audio. It is nominally ahead of the film-average embedding ($+0.023$, Nadeau--Bengio $p=0.536$) and slightly behind the vote over cues. None of these differences is significant, and the PCA-8 control is close to all of them.

76. **Before:** The emotion pipeline reaches 0.508 against a random-guess floor of 0.292. That is about 82\,\% of the 0.620 reached with in-domain training,  
   **After:** The emotion pipeline reaches 0.508, about 82\,\% of the score reached with in-domain training,

77. **Before:** It outperforms direct VGGish (0.403) by $+0.106$ [$+0.021$, $+0.184$] ($p=0.013$) and the PCA-8 control (0.417) by $+0.093$  
   **After:** It outperforms direct VGGish by $+0.106$ [$+0.021$, $+0.184$] ($p=0.013$) and the PCA-8 control by $+0.093$

78. **Before:** The emotion pipeline recovers more of the true genres (0.651 against 0.435), at lower precision (0.480 against 0.705).  
   **After:** The emotion pipeline recovers more of the true genres, at a lower precision (Table~\ref{tab:zero_shot}).

79. **Before:** the zero-shot transfer was repeated with a separate threshold per genre.  
   **After:** I repeated the zero-shot transfer with a separate threshold per genre.

80. **Before:** The emotion pipeline is ahead of direct VGGish in both transfer directions, but significantly only from Eerola to Blockbuster (0.503 against 0.383, $+0.121$, $p=0.007$). From Blockbuster to Eerola it reaches 0.458 against 0.364 ($+0.089$ [$-0.017$, $+0.206$], $p=0.102$), against a random-guess floor of 0.240.  
   **After:** The emotion pipeline is ahead of direct VGGish in both transfer directions, but significantly only from Eerola to Blockbuster ($+0.121$, $p=0.007$). From Blockbuster to Eerola its lead is $+0.089$ [$-0.017$, $+0.206$] ($p=0.102$).

81. **Before:** Per film, the emotion pipeline (0.531) and direct VGGish (0.543) are on par ($-0.011$, Nadeau--Bengio $p=0.688$).  
   **After:** Per film, the emotion pipeline and direct VGGish perform similarly ($-0.011$, Nadeau--Bengio $p=0.688$).

82. **Before:** It was fixed first, and it carries the further checks:  
   **After:** I fixed it first, and it carries the further checks:

83. **Before:** This was checked against the IMDb identifiers of all films.  
   **After:** I checked this against the IMDb identifiers of all films.

84. **Before:** All Eerola experiments were re-run with the two spellings merged.  
   **After:** I re-ran all Eerola experiments with the two spellings merged.

85. **Before:** reaches a film-level macro-$F_1$ of 0.460, against 0.309 for a random guess. The human ratings reach 0.491.  
   **After:** reaches a film-level macro-$F_1$ of 0.460, well above a random guess, and the human ratings reach 0.491.

86. **Before:** given directly to the same classifier (by $+0.093$ over VGGish, its own input) and of the PCA-8 control,  
   **After:** given directly to the same classifier, including VGGish, its own input, and of the PCA-8 control,

87. **Before:** direct VGGish at the default threshold (0.508 against 0.403, $p=0.013$).  
   **After:** direct VGGish at the default threshold.

88. **Before:** and with both datasets in training the pipelines are on par.  
   **After:** and with both datasets in training the pipelines reach similar scores.

## Chapter 6: Discussion (5 changes)

89. **Before:** gives a genre result on par with VGGish  
   **After:** gives a genre result comparable to that of VGGish

90. **Before:** and with both datasets in training the pipelines are on par.  
   **After:** and with both datasets in training the pipelines reach similar scores.

91. **Before:** It is on par with direct audio where the classifier was trained  
   **After:** It performs about as well as direct audio where the classifier was trained

92. **Before:** Scoring per film was adopted after the clip-level analysis had been done.  
   **After:** I adopted scoring per film after the clip-level analysis had been done.

93. **Before:** The five headline genres were chosen partly because they carry  
   **After:** I chose the five headline genres partly because they carry

## Chapter 7: Conclusions (3 changes)

94. **Before:** This thesis examined how well  
   **After:** In this thesis I examined how well

95. **Before:** and with both datasets in training the pipelines are on par.  
   **After:** and with both datasets in training the two pipelines perform similarly.

96. **Before:** Some decisions of this thesis were made after the first results were known. Film-level scoring was adopted after the clip-level analysis, and the five-genre subset was chosen partly because  
   **After:** I made some decisions after the first results were known. I adopted film-level scoring after the clip-level analysis, and I chose the five-genre subset partly because

## Appendix A (3 changes)

97. **Before:** This is on par with the VGGish pipeline  
   **After:** This is comparable to the VGGish pipeline

98. **Before:** It is on par on Horror  
   **After:** It is close to it on Horror

99. **Before:** Tuning on the 319 clips was fixed as the main variant before the run,  
   **After:** I fixed tuning on the 319 clips as the main variant before the run,
