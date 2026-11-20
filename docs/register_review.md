# Academic register: every rewritten phrase

After the supervisor meeting of 12 Nov 2026 the thesis was revised for academic register: colloquial verbs ("beats", "gets", "squashed", "says yes"), rhetorical questions, sentence-initial "And/But/So" and informal phrases were replaced, keeping sentences short. A script checked that every number, citation, reference and label is unchanged in each chapter. Each pair below is *before* / *after*, in the LaTeX source; check them and say which to undo or rephrase (quote the number).

Not listed here, because they are new text rather than rephrasing: Section 1.1 Motivation (rewritten in the CARS structure) and Chapter 7 (first draft).

## Chapter 1, Introduction (Sections 1.2 and 1.3) (8 changes)

1. **Before:** The aim is to find out how well soundtrack emotion works as an interpretable step between film audio and film genre.  
   **After:** The aim is to determine how well soundtrack emotion serves as an interpretable step between film audio and film genre.

2. **Before:** RQ2 is about the shape of the signal: which emotions set each genre apart, and whether the same emotional signatures turn up in a second collection of films. This checks whether the emotional step can be read the way the motivation above suggests.  
   **After:** RQ2 concerns the form of the signal: which emotions distinguish each genre, and whether the same emotional signatures appear in a second collection of films. This tests whether the emotional step can be interpreted in the way the motivation above suggests.

3. **Before:** RQ4 asks how the two pipelines do when a model trained on one collection of films is applied to another without seeing any of its labels.  
   **After:** RQ4 asks how the two pipelines perform when a model trained on one collection of films is applied to another without access to any of its labels.

4. **Before:** both use the second dataset, but they ask different things.  
   **After:** both use the second dataset, but they address different questions.

5. **Before:** a control that squeezes the representation into as many numbers as there are emotions, without giving those numbers any emotional meaning.  
   **After:** a control that compresses the representation into as many values as there are emotions, without giving those values any emotional meaning.

6. **Before:** the two learning methods and the ways the results are measured.  
   **After:** the two learning methods and the evaluation methods.

7. **Before:** the baselines and controls, and how the evaluation is done.  
   **After:** the baselines and controls, and the evaluation protocol.

8. **Before:** Chapter~\ref{ch:conclusions} sums up the answers and names the next steps.  
   **After:** Chapter~\ref{ch:conclusions} summarises the answers and outlines directions for future work.

## Chapter 2, Fundamentals (72 changes)

9. **Before:** So why does this thesis use eight emotion dimensions (valence, energy arousal, tension arousal, anger, fear, happiness, sadness, tenderness) and not just the two of the circumplex model? Three things bear on the choice, and only some of them are strictly necessary.  
   **After:** This thesis uses eight emotion dimensions (valence, energy arousal, tension arousal, anger, fear, happiness, sadness, tenderness) rather than only the two of the circumplex model. Three considerations bear on this choice, and not all of them are strictly necessary.

10. **Before:** Anger and fear share the unpleasant, high-arousal quarter,  
   **After:** Anger and fear share the unpleasant, high-arousal quadrant,

11. **Before:** which is added here in the pleasant, energetic quarter.  
   **After:** which is added here in the pleasant, energetic quadrant.

12. **Before:** One reason is that it is fairly easy to annotate  
   **After:** One reason is that it is comparatively easy to annotate

13. **Before:** Discrete categories are the more diagnostic description -- but not a necessary one.  
   **After:** Discrete categories are the more diagnostic description, but not a necessary one.

14. **Before:** Anger does not tell them apart at all  
   **After:** Anger does not distinguish them at all

15. **Before:** Fear is thus the most telling single emotion, and ``Horror is fear'' says more about what a score is doing than ``Horror is low valence and high tension''.  
   **After:** Fear is thus the most informative single emotion, and the description ``Horror is fear'' says more about the function of a score than ``Horror is low valence and high tension''.

16. **Before:** subsets of the eight emotions, and it does not need them.  
   **After:** subsets of the eight emotions, and the classifier does not depend on them.

17. **Before:** so a small subset already holds most of what matters for genre.  
   **After:** so a small subset already contains most of the information relevant to genre.

18. **Before:** All eight are kept for other reasons than prediction. The discrete categories are the ones that name what a score is doing, so they make the model easier to read. Nobody can know beforehand which subset would be enough. And eleven features cost almost nothing to compute and to model.  
   **After:** All eight are kept for reasons other than prediction. The discrete categories name the function of a score directly, so they make the model easier to interpret. It cannot be known in advance which subset would suffice. Finally, eleven features add almost no cost to computation or modelling.

19. **Before:** and that is what this thesis cares about.  
   **After:** and this is the focus of this thesis.

20. **Before:** It also reports that this cut would have cost nothing in accuracy.  
   **After:** It also reports that this reduction would not have lowered accuracy.

21. **Before:** This is more than a curious pattern in the data.  
   **After:** This is more than a statistical regularity in the data.

22. **Before:** Composers use these codes on purpose,  
   **After:** Composers use these codes deliberately,

23. **Before:** Driving, dissonant, tense cues, for example, are the stock sound of horror and action.  
   **After:** Driving, dissonant and tense cues, for example, are conventional in horror and action scores.

24. **Before:** Seen this way, emotion is more than a handy intermediate step  
   **After:** From this perspective, emotion is more than a convenient intermediate step

25. **Before:** A learning algorithm cannot work on an audio file as it is.  
   **After:** A learning algorithm cannot operate on an audio file directly.

26. **Before:** Everything after this step works on these vectors,  
   **After:** All later steps operate on these vectors,

27. **Before:** The frequency axis is bent onto the \emph{mel} scale  
   **After:** The frequency axis is mapped onto the \emph{mel} scale

28. **Before:** because we hear loudness that way.  
   **After:** because perceived loudness grows roughly logarithmically with energy.

29. **Before:** In doing so it keeps roughly what the ear keeps, which is why it works well.  
   **After:** It thereby retains roughly the information that the ear retains, which is one reason for its wide use.

30. **Before:** and then summed up over time by its mean and standard deviation.  
   **After:** and then summarised over time by its mean and standard deviation.

31. **Before:** Their weakness is that they only capture what their designers thought of measuring.  
   **After:** Their weakness is that they capture only the properties their designers chose to measure.

32. **Before:** A network that has heard millions of sounds tends to encode things like instrumentation,  
   **After:** A network trained on millions of sounds tends to encode properties such as instrumentation,

33. **Before:** It is included because it is different from the rest. If the other four agree, it helps to know whether that is only because they share the mel front-end.  
   **After:** It is included because it differs from the others. If the other four agree, it shows whether this agreement is only due to their shared mel front-end.

34. **Before:** Dimension 87 of a VGGish vector encodes something. But nothing guarantees it encodes the same thing in another dataset, and no person can read it.  
   **After:** Dimension 87 of a VGGish vector encodes some property of the audio. Nothing guarantees, however, that it encodes the same property in another dataset, and a person cannot interpret it.

35. **Before:** Whether it does is the only thing worth measuring,  
   **After:** Whether it does is what the evaluation has to establish,

36. **Before:** the simplest sound approach.  
   **After:** the simplest principled approach.

37. **Before:** One independent yes/no classifier is trained per genre, and their answers are put side by side.  
   **After:** One independent binary classifier is trained per genre, and their outputs are combined into one label vector.

38. **Before:** The weighted sum of the features is squashed into a probability, and the clip gets the genre if this probability is above one half.  
   **After:** The logistic function maps the weighted sum of the features to a probability, and the clip is assigned the genre if this probability exceeds one half.

39. **Before:** Most important here, it can be read:  
   **After:** Most importantly here, it is interpretable:

40. **Before:** It does so by latching onto chance correlations that will not come back in new data. It then memorises instead of learning.  
   **After:** It does so by fitting chance correlations that do not recur in new data. The model then memorises the training data instead of learning a general relation.

41. **Before:** which pulls it towards simpler solutions.  
   **After:** which favours simpler solutions.

42. **Before:** was far too weak for every feature set,  
   **After:** proved far too weak for every feature set,

43. **Before:** The default hurt the large embeddings,  
   **After:** The default harmed the large embeddings,

44. **Before:** Each tree splits the input space by asking questions like  
   **After:** Each tree splits the input space by tests such as

45. **Before:** A bottleneck throws information away, so it can only help if what it keeps is worth more than what it loses. This thesis claims that this bottleneck keeps what matters for genre and drops what does not. It also claims that it does so  
   **After:** A bottleneck discards information, so it can only help if the information it retains is more useful than the information it discards. This thesis hypothesises that this bottleneck retains the information relevant to genre and discards the rest. It further hypothesises that it does so

46. **Before:** The second claim can be tested.  
   **After:** The second hypothesis can be tested.

47. **Before:** \section{Measuring Whether It Works}  
   **After:** \section{Measuring Model Performance}

48. **Before:** How well a model does on its own training data says nothing about how it generalises. The standard answer is  
   **After:** A model's performance on its own training data says little about how it generalises. The standard remedy is

49. **Before:** On this dataset a naive split is wrong, and the error can be measured.  
   **After:** On this dataset a naive split is invalid, and the resulting error can be measured.

50. **Before:** If two clips from \emph{Gladiator} land on opposite sides of a split,  
   **After:** If one clip from \emph{Gladiator} is in the training set and another in the test set,

51. **Before:** That is far too few to tell a real difference of a few hundredths from a lucky split,  
   **After:** These are too few to distinguish a real difference of a few hundredths from a favourable split,

52. **Before:** Last, each repeat is scored once,  
   **After:** Finally, each repeat is scored once,

53. **Before:** however good the model is.  
   **After:** regardless of the quality of the model.

54. **Before:** $R^2 = 0$ is what always guessing the average rating gets,  
   **After:** $R^2 = 0$ corresponds to always predicting the mean rating,

55. **Before:** One reference point is needed to read it.  
   **After:** Its interpretation requires one reference point.

56. **Before:** The Eerola dataset lets us measure this noise,  
   **After:** The Eerola dataset allows this noise to be measured,

57. **Before:** It is the headline measure because the alternatives mislead badly with labels as unbalanced as these.  
   **After:** It is the primary measure because the alternatives are misleading when labels are as imbalanced as these.

58. **Before:** Both beat every real model (at best 0.168 and 0.311).  
   **After:** Both values are better than those of every trained model (at best 0.168 and 0.311).

59. **Before:** Macro-$F_1$ cannot be fooled this way. A genre that a classifier never predicts gets an $F_1$ of zero and drags the average down.  
   **After:** Macro-$F_1$ does not have this weakness. A genre that a classifier never predicts receives an $F_1$ of zero, which lowers the average.

60. **Before:** The classifier still works on clips. For scoring, though,  
   **After:** The classifier still operates on clips. For scoring, however,

61. **Before:** The Blockbuster dataset has one row per film anyway, so this way every genre result  
   **After:** The Blockbuster dataset already has one row per film, so in this way every genre result

62. **Before:** so the guess says Drama with a probability of 0.725.  
   **After:** so the baseline assigns Drama with a probability of 0.725.

63. **Before:** so it says Comedy with a probability of 0.10. The guess knows nothing about a film,  
   **After:** so the baseline assigns Comedy with a probability of 0.10. The baseline uses no information about a film,

64. **Before:** because it picks that share of the films  
   **After:** because it selects that share of the films

65. **Before:** makes a result look better by more than a tenth of an $F_1$ point.  
   **After:** overstates a result by more than a tenth of an $F_1$ point.

66. **Before:** For one genre it answers a simple question. If one film with the genre and one film without it are picked at random, how often does the film with the genre get the higher probability? A value of 1.0 means always, so the ranking is perfect. A value of 0.5 means half the time, which is what a coin would do.  
   **After:** For one genre, it is the probability that a randomly chosen film with the genre receives a higher predicted probability than a randomly chosen film without it. A value of 1.0 means that this holds for every such pair, so the ranking is perfect. A value of 0.5 means that it holds for half of the pairs, which is what a random ordering achieves on average.

67. **Before:** Film A beats all three others. Film C beats D and E but falls behind B.  
   **After:** Film A is ranked above all three others. Film C is ranked above D and E but below B.

68. **Before:** and equally for one that always says ``Drama''.  
   **After:** and equally for one that always predicts ``Drama''.

69. **Before:** It shows whether a model puts the right films first,  
   **After:** It shows whether a model ranks the correct films first,

70. **Before:** Several analyses do not test a model. They ask whether  
   **After:** Several analyses do not evaluate a model but test whether

71. **Before:** the genre being looked at,  
   **After:** the genre under consideration,

72. **Before:** Two tests are used to judge it,  
   **After:** Two tests are used to assess it,

73. **Before:** and the difference is noted.  
   **After:** and the difference is recorded.

74. **Before:** The lead is likely real, but another sample of films could still remove it.  
   **After:** The lead is probably real, but another sample of films could still eliminate it.

75. **Before:** But the folds share most of their training films,  
   **After:** However, the folds share most of their training films,

76. **Before:** So is the number of the ten repeats in which one model beats the other, which needs no assumptions at all.  
   **After:** The number of the ten repeats in which one model outperforms the other is reported as well; it requires no distributional assumptions.

77. **Before:** With 18 tests and no real difference anywhere, about one would.  
   **After:** With 18 tests and no real difference, about one such result is expected.

78. **Before:** The discrete categories pull the two genres apart along one axis.  
   **After:** The discrete categories separate the two genres along one axis.

79. **Before:** and the film gets a genre when that average is at least one half.  
   **After:** and the film is assigned a genre when that average is at least one half.

80. **Before:** so that clips that sound alike end up close together.  
   **After:** so that clips that sound alike lie close together.

## Chapter 3, Related Work (35 changes)

81. **Before:** The second group is four studies that classify film genre  
   **After:** The second group comprises four studies that classify film genre

82. **Before:** The finding that matters most here is that arousal  
   **After:** The finding most relevant here is that arousal

83. **Before:** several kinds of features used together tend to beat any single kind on its own.  
   **After:** several kinds of features used together tend to outperform any single kind on its own.

84. **Before:** is still there.  
   **After:** persists.

85. **Before:** as a way around the small size of emotion datasets. This thesis does the same. It uses such encoders, frozen, to extract features.  
   **After:** as a way to compensate for the small size of emotion datasets. This thesis follows that suggestion and uses such encoders, frozen, to extract features.

86. **Before:** The last point leads to one of the main ideas of this thesis.  
   **After:** The last point motivates one of the main ideas of this thesis.

87. **Before:** The thesis also asks which emotions drive which genre predictions.  
   **After:** The thesis also examines which emotions drive which genre predictions.

88. **Before:** It is an old finding in music information retrieval, it has been repeated by independent groups,  
   **After:** It is an established finding in music information retrieval, it has been replicated by independent groups,

89. **Before:** tested the other direction: do models of musical emotion themselves depend on genre?  
   **After:** tested the opposite direction, namely whether models of musical emotion themselves depend on genre.

90. **Before:** So emotion and genre are tied closely enough that  
   **After:** Emotion and genre are thus related closely enough that

91. **Before:** It also says what to expect when a model moves to a new dataset.  
   **After:** It also indicates what to expect when a model is applied to a new dataset.

92. **Before:** turn this dependence to good use. Their mood models that adapt to the genre beat the same models  
   **After:** exploit this dependence. Their genre-adaptive mood models outperform the same models

93. **Before:** show that the link between emotion and genre runs both ways and is reliable.  
   **After:** show that the link between emotion and genre holds in both directions and is replicable.

94. **Before:** were among the first to predict film genre straight from soundtrack audio.  
   **After:** were among the first to predict film genre directly from soundtrack audio.

95. **Before:** are easier to tell apart by their music than calmer, story-driven ones  
   **After:** are easier to distinguish by their music than calmer, narrative-driven ones

96. **Before:** The same pattern shows up in the emotional signatures  
   **After:** The same pattern appears in the emotional signatures

97. **Before:** The authors admit that giving each film one genre is probably too coarse,  
   **After:** The authors acknowledge that assigning each film a single genre is probably too coarse,

98. **Before:** First, \citeauthor{austin2010characterization} give each film one genre.  
   **After:** First, \citeauthor{austin2010characterization} assign each film a single genre.

99. **Before:** as an explicit step that can be read, instead of classifying genre straight from low-level audio features.  
   **After:** as an explicit, interpretable step, instead of classifying genre directly from low-level audio features.

100. **Before:** Several tracks of one film can then end up on both sides of a cross-validation split.  
   **After:** Several tracks of one film can then appear on both sides of a cross-validation split.

101. **Before:** It finds out automatically which tracks of a soundtrack are actually used in the film  
   **After:** It determines automatically which tracks of a soundtrack are actually used in the film

102. **Before:** report that models on VGGish embeddings generally beat those on the hand-crafted MIR features.  
   **After:** report that models on VGGish embeddings generally outperform those on the hand-crafted MIR features.

103. **Before:** The learned embedding and the full hand-crafted feature set cannot be told apart statistically  
   **After:** The learned embedding and the full hand-crafted feature set do not differ significantly

104. **Before:** That gives a more direct and controlled link between  
   **After:** This provides a more direct and controlled link between

105. **Before:** The other looks at trailer frames with  
   **After:** The other processes trailer frames with

106. **Before:** What matters most here is that audio on its own was the weakest  
   **After:** Most relevant here is that audio on its own was the weakest

107. **Before:** This thesis does not take this as evidence against using audio. The weakness of raw audio is in fact what its central hypothesis is about.  
   **After:** This thesis does not treat this result as evidence against using audio. The weakness of raw audio is, in fact, the subject of its central hypothesis.

108. **Before:** Still, \citeauthor{behrouzi2023multimodal} support one choice made here on their own:  
   **After:** Nevertheless, \citeauthor{behrouzi2023multimodal} independently support one choice made here:

109. **Before:** Their headline metric, the Hamming loss, is not used here, though. With the class imbalance in the dataset of this thesis, a trivial classifier gets a better Hamming loss than any real model.  
   **After:** Their headline metric, the Hamming loss, is not used here, however. With the class imbalance in the dataset of this thesis, a trivial classifier achieves a better Hamming loss than any trained model.

110. **Before:** The models therefore give no hint of why a genre was predicted. The present work fills this gap.  
   **After:** The models therefore give no indication of why a genre was predicted. The present work addresses this gap.

111. **Before:** or audio alone predicts genre fairly poorly. Both point to an emotion step as a possible answer, rather than an ever larger audio model.  
   **After:** or audio alone predicts genre poorly. Both problems suggest an emotion step as a possible remedy, rather than an ever larger audio model.

112. **Before:** so tracks of one film can land in both the training and the test folds.  
   **After:** so tracks of one film can appear in both the training and the test folds.

113. **Before:** and it also measures what this safeguard is worth. The same audio baseline scores clearly higher on an ungrouped split than on a grouped one. Much of its apparent skill is thus recognising the film,  
   **After:** and it also quantifies the effect of this safeguard. The same audio baseline scores markedly higher on an ungrouped split than on a grouped one. Much of its apparent performance thus comes from recognising the film,

114. **Before:** The first question can be asked more directly: are the discrete emotions needed at all, since they can be placed in the plane of valence and arousal? Section~\ref{subsec:fund_why_not_va} answers it.  
   **After:** The first question can be posed more directly, namely whether the discrete emotions are needed at all, since they can be placed in the plane of valence and arousal. Section~\ref{subsec:fund_why_not_va} addresses it.

115. **Before:** The eight are kept because the discrete categories name what a score is doing.  
   **After:** The eight are kept because the discrete categories name the function of a score.

## Chapter 4, Methods (61 changes)

116. **Before:** The other is a control that squeezes the representation into eight dimensions  
   **After:** The other is a control that compresses the representation into eight dimensions

117. **Before:** Chapter~\ref{ch:fundamentals} explains the ideas behind them; this chapter says what was done.  
   **After:** Chapter~\ref{ch:fundamentals} explains the underlying concepts; this chapter describes how they were applied.

118. **Before:** it is as narrow as the emotion pipeline, but its numbers mean nothing emotional.  
   **After:** it is as narrow as the emotion pipeline, but its values have no emotional meaning.

119. **Before:** They say what the listeners judged the music to express, not what they felt.  
   **After:** They describe what the listeners judged the music to express, not what they felt.

120. **Before:** Here the claim that emotion generalises is put to the test.  
   **After:** This setting tests the claim that emotion generalises.

121. **Before:** The other order would cause a problem. If the embeddings were averaged over the film first and emotion predicted only once, the regressor would get a much smoother vector than anything it saw in training.  
   **After:** The reverse order would be problematic. If the embeddings were averaged over the film first and emotion predicted only once, the regressor would receive a much smoother vector than any it encountered in training.

122. **Before:** Against this, eight genres from 42 soundtracks is a lot, and three of them have little support.  
   **After:** Compared with these studies, eight genres from 42 soundtracks is a large number, and three of them have little support.

123. **Before:** The five-genre results show how the approach performs where it can work,  
   **After:** The five-genre results show how the approach performs where it is applicable,

124. **Before:** A model can only move between two datasets if it predicts the same classes in both. The two lists of genres differ, though.  
   **After:** A model can only be transferred between two datasets if it predicts the same classes in both. The two lists of genres differ, however.

125. **Before:** They do occur in the IMDb genre lists of some Eerola films, though:  
   **After:** They do, however, occur in the IMDb genre lists of some Eerola films:

126. **Before:** Six representations are compared, so that a conclusion about ``direct audio'' does not rest on one model that may be a poor choice.  
   **After:** Six representations are compared, so that a conclusion about ``direct audio'' does not depend on a single, possibly unsuitable model.

127. **Before:** Table~\ref{tab:representations} lists what was used.  
   **After:** Table~\ref{tab:representations} lists the representations used.

128. **Before:** so the shorter windows throw part of the audio away. This was tested.  
   **After:** so the shorter windows discard part of the audio. The effect of this was tested.

129. **Before:** It made no real difference to either emotion or genre prediction.  
   **After:** The pooling method made no substantial difference to either emotion or genre prediction.

130. **Before:** Pooling the last hidden state, the obvious default, may therefore be the wrong choice for music.  
   **After:** Pooling the last hidden state, the usual default, may therefore be unsuitable for music.

131. **Before:** The first three transformer blocks do best  
   **After:** The first three transformer blocks perform best

132. **Before:** The last five blocks fall to between 0.154 and 0.170  
   **After:** The last five blocks drop to between 0.154 and 0.170

133. **Before:** and the final layer would have made the waveform representation look much worse than it is.  
   **After:** and the final layer would have understated the quality of the waveform representation considerably.

134. **Before:** A weak result for it therefore cannot be blamed on the waveform input alone.  
   **After:** A weak result for it therefore cannot be attributed to the waveform input alone.

135. **Before:** A first comparison used the default and so made the random forest look better than it is.  
   **After:** A first comparison used the default and therefore overstated the advantage of the random forest.

136. **Before:** Picking the best of its 18 entries afterwards would favour whichever entry happened to do well on these 40 films.  
   **After:** Selecting the best of its 18 entries afterwards would favour whichever entry happened to perform well on these 40 films.

137. **Before:** Third, the random forest needs no tuning of its own and is the safest choice across inputs.  
   **After:** Third, the random forest needs no tuning of its own and is the most robust choice across inputs.

138. **Before:** but a more accurate emotion model does not reliably give a better genre result  
   **After:** but a more accurate emotion model does not reliably yield a better genre result

139. **Before:** The two composites sum up the negative and the positive emotions in one number each.  
   **After:** The two composites summarise the negative and the positive emotions in one value each.

140. **Before:** these predictions must be as noisy as the ones it gets at test time.  
   **After:** these predictions must be as noisy as the ones it receives at test time.

141. **Before:** but the out-of-fold procedure is the right one, and it is the one reported.  
   **After:** but the out-of-fold procedure is the correct one, and it is the one reported.

142. **Before:** A clip gets a genre when the predicted probability is above $0.5$.  
   **After:** A clip is assigned a genre when the predicted probability exceeds $0.5$.

143. **Before:** Ties go to the smaller value,  
   **After:** Ties are resolved in favour of the smaller value,

144. **Before:** It also hurt the large embeddings more than the eleven emotion features,  
   **After:** It also harmed the large embeddings more than the eleven emotion features,

145. **Before:** None of them did better than the regularised logistic regression, and gradient boosting overfitted clearly.  
   **After:** None of them outperformed the regularised logistic regression, and gradient boosting overfitted markedly.

146. **Before:** Each one answers a specific question.  
   **After:** Each one addresses a specific question.

147. **Before:** so a positive result cannot be explained away by a weak baseline.  
   **After:** so a positive result cannot be attributed to a weak baseline.

148. **Before:** The emotion pipeline squeezes the audio into eight numbers. On a small dataset, any strong compression might help just by acting as a regulariser.  
   **After:** The emotion pipeline compresses the audio into eight values. On a small dataset, any strong compression might help merely by acting as a regulariser.

149. **Before:** If the emotion pipeline beat the direct embedding only because it is small, the PCA-8 control would beat it as well. The control is as narrow as the emotion pipeline. But its numbers have no emotional meaning,  
   **After:** If the emotion pipeline outperformed the direct embedding only because it is small, the PCA-8 control would outperform it as well. The control is as narrow as the emotion pipeline, but its values have no emotional meaning,

150. **Before:** It shows how much of what eight dimensions can carry depends on choosing them well, and so whether PCA-8 is a serious competitor or only a straw man.  
   **After:** It shows how much of the information that eight dimensions can carry depends on choosing them well, and thus whether PCA-8 is a serious competitor or only a weak reference.

151. **Before:** It is the floor for exact match and Hamming loss, where it is hard to beat.  
   **After:** It is the floor for exact match and Hamming loss, where it is difficult to outperform.

152. **Before:** This makes it the higher floor for macro-$F_1$, and the better one.  
   **After:** This makes it the higher floor for macro-$F_1$, and the more appropriate one.

153. **Before:** Chapter~\ref{ch:evaluation} says for each score which of the two floors is meant.  
   **After:** Chapter~\ref{ch:evaluation} states for each score which of the two floors is meant.

154. **Before:** but it shows how much of a macro-$F_1$ score recall alone can buy  
   **After:** but it shows how much of a macro-$F_1$ score can be achieved through recall alone

155. **Before:** This safeguard is needed.  
   **After:** This safeguard is necessary.

156. **Before:** which cannot tell apart differences of a few hundredths of macro-$F_1$.  
   **After:** which cannot distinguish differences of a few hundredths of macro-$F_1$.

157. **Before:** The film gets every genre whose average probability is at least $0.5$.  
   **After:** The film is assigned every genre whose average probability is at least $0.5$.

158. **Before:** A single film therefore changes the $F_1$ of such a genre a great deal.  
   **After:** A single film therefore changes the $F_1$ of such a genre considerably.

159. **Before:** This way no test clip enters either fit.  
   **After:** In this way, no test clip enters either fit.

160. **Before:** This asks whether the two datasets can be learned from together, which neither direction of transfer does.  
   **After:** This design tests whether a model can learn from both datasets together, which neither direction of transfer does.

161. **Before:** A second-stage classifier does not get in the way of this measure.  
   **After:** This measure is not affected by a second-stage classifier.

162. **Before:** rewards a model for focusing on the frequent genres. For example, a classifier that predicts ``Drama'' for every clip reaches a better Hamming loss than any real model tested here. It also reaches an exact-match ratio that none of them beats, although it is useless.  
   **After:** rewards a model for focusing on the frequent genres. For example, a classifier that predicts ``Drama'' for every clip achieves a better Hamming loss than any trained model tested here. It also achieves an exact-match ratio that none of them exceeds, although it has no practical use.

163. **Before:** They also show whether a model errs by predicting too many genres or too few, which $F_1$ alone does not.  
   **After:** They also show whether a model errs by predicting too many genres or too few, which $F_1$ alone does not reveal.

164. **Before:** Every metric that needs a yes or no can be fooled by a classifier that has learned nothing. Different metrics are fooled by different classifiers. One that says only ``Drama'' wins on Hamming loss and exact match, because it makes few false alarms. One that says yes to every genre does well on macro-$F_1$, because $F_1$ rewards recall.  
   **After:** Every metric that requires a binary decision can be satisfied by a classifier that has learned nothing, and different metrics favour different trivial classifiers. One that predicts only ``Drama'' scores best on Hamming loss and exact match, because it makes few false alarms. One that assigns every genre scores well on macro-$F_1$, because $F_1$ rewards recall.

165. **Before:** ROC-AUC closes this gap. It only asks whether items with a genre get higher probabilities than items without it. Both trivial classifiers give every item the same answer,  
   **After:** ROC-AUC closes this gap. It only measures whether items with a genre receive higher probabilities than items without it. Both trivial classifiers give every item the same output,

166. **Before:** ROC-AUC alone is not enough either. It says nothing about the decisions a model makes,  
   **After:** ROC-AUC alone is not sufficient either. It provides no information about the decisions a model makes,

167. **Before:** A lead in $F_1$ alone may only come from saying yes more often.  
   **After:** A lead in $F_1$ alone may only come from assigning genres more often.

168. **Before:** because ROC-AUC can look too optimistic when the classes are unbalanced.  
   **After:** because ROC-AUC can be overly optimistic when the classes are unbalanced.

169. **Before:** and the classifier that says yes to everything only reaches the share of clips with each genre.  
   **After:** and the all-positive classifier only reaches the share of clips with each genre.

170. **Before:** and a film-level score per fold means little.  
   **After:** and a film-level score per fold is of little value.

171. **Before:** Only one term of the corrected variance shrinks with more repeats. So if $p_{\lim} > 0.05$,  
   **After:** Only one term of the corrected variance shrinks with more repeats. Hence, if $p_{\lim} > 0.05$,

172. **Before:** To find out which emotions go with which genres,  
   **After:** To determine which emotions are associated with which genres,

173. **Before:** so plain accuracy is the right measure there,  
   **After:** so plain accuracy is the appropriate measure there,

174. **Before:** so that any difference can be put down to the correction alone.  
   **After:** so that any difference can be attributed to the correction alone.

175. **Before:** goes into the Stage~2 classifier without any step in between.  
   **After:** is passed to the Stage~2 classifier without any step in between.

176. **Before:** The result goes into the same classifier on the same folds.  
   **After:** The result is passed to the same classifier on the same folds.

## Chapter 5, Experiments and Evaluation (43 changes)

177. **Before:** genre. Section~\ref{sec:eval_signatures} looks at which emotions go with which genres.  
   **After:** genre. Section~\ref{sec:eval_signatures} examines which emotions are associated with which genres.

178. **Before:** Section~\ref{sec:eval_transfer} moves to the second dataset and tests transfer between the two. Section~\ref{sec:eval_summary} sums up the results.  
   **After:** Section~\ref{sec:eval_transfer} turns to the second dataset and tests transfer between the two. Section~\ref{sec:eval_summary} summarises the results.

179. **Before:** Third, macro-$F_1$ can be pushed up by saying yes more often.  
   **After:** Third, macro-$F_1$ can be raised by assigning genres more often.

180. **Before:** AST, cannot be told apart ($R^2$ between 0.545 and 0.558).  
   **After:** AST, do not differ substantially ($R^2$ between 0.545 and 0.558).

181. **Before:** panel. The panels agree closely  
   **After:** panel. The two panels agree closely

182. **Before:** On the three bipolar scales, though, the second panel rated systematically higher,  
   **After:** On the three bipolar scales, however, the second panel rated systematically higher,

183. **Before:** On the human ratings the classifier reaches  
   **After:** With the human ratings as input, the classifier reaches

184. **Before:** Predicted emotions lose little against the ratings.  
   **After:** Predicted emotions perform only slightly worse than the ratings.

185. **Before:** Over clips the two are level ($-0.010$,  
   **After:** Over clips the two are on par ($-0.010$,

186. **Before:** This experiment tests whether the emotion step helps.  
   **After:** This experiment tests whether the emotion step improves genre prediction.

187. **Before:** compresses the VGGish embedding to eight numbers without any emotional meaning  
   **After:** compresses the VGGish embedding to eight values without any emotional meaning

188. **Before:** Second, compression alone gives part of the gain at film level.  
   **After:** Second, compression alone accounts for part of the gain at film level.

189. **Before:** Over clips the control is level with VGGish,  
   **After:** Over clips the control is on par with VGGish,

190. **Before:** the pipelines come together over clips (0.420 to 0.433;  
   **After:** the pipelines converge over clips (0.420 to 0.433;

191. **Before:** ($p=0.093$). That is about as large as its lead in $F_1$, but not significant either.  
   **After:** ($p=0.093$). This is about as large as its lead in $F_1$, but it is not significant either.

192. **Before:** These numbers are stable. Re-running the whole protocol under five master seeds moves the  
   **After:** These values are stable. Re-running the whole protocol under five master seeds changes the

193. **Before:** Second, inside the emotion pipeline the representation hardly matters.  
   **After:** Second, within the emotion pipeline the choice of representation has little effect.

194. **Before:** A more accurate emotion model does not reliably give a better genre result.  
   **After:** A more accurate emotion model does not reliably yield a better genre result.

195. **Before:** On MusiCNN it predicts them less well and scores higher.  
   **After:** On MusiCNN it predicts them less accurately and scores higher.

196. **Before:** A post-hoc test does not show this lead to be real, however:  
   **After:** A post-hoc test, however, does not confirm this lead:

197. **Before:** VGGish used directly seldom predicts it (recall 0.12), whereas the emotion pipeline finds most Comedy films  
   **After:** VGGish used directly rarely predicts it (recall 0.12), whereas the emotion pipeline identifies most Comedy films

198. **Before:** the genre for too many films (precision 0.15 at a recall of 0.50) and is no better than VGGish  
   **After:** the genre for too many films (precision 0.15 at a recall of 0.50) and performs no better than VGGish

199. **Before:** Only four or five films stand behind the two rarest genres, and their $F_1$ values move by about 0.2 when a single film changes.  
   **After:** The two rarest genres rest on only four or five films, and their $F_1$ values change by about 0.2 when a single film changes.

200. **Before:** This analysis looks at what the link between emotion and genre looks like, not at how well it predicts.  
   **After:** This analysis describes the form of the link between emotion and genre, not its predictive strength.

201. **Before:** The second dataset can test whether the signatures describe film music in general or only the Eerola films.  
   **After:** The second dataset makes it possible to test whether the signatures describe film music in general or only the Eerola films.

202. **Before:** The absolute effect sizes do not transfer. Averaging some 39 cues per film inflates $d$,  
   **After:** The absolute effect sizes do not transfer. Averaging about 39 cues per film inflates $d$,

203. **Before:** In-domain on Blockbuster, the emotion pipeline (0.616 $\pm$ 0.022) is level with direct audio.  
   **After:** In-domain on Blockbuster, the emotion pipeline (0.616 $\pm$ 0.022) is on par with direct audio.

204. **Before:** The PCA-8 control (0.587) is level with them as well.  
   **After:** The PCA-8 control (0.587) is on par with them as well.

205. **Before:** A film gets a genre when more than half of its cues predict it.  
   **After:** A film is assigned a genre when more than half of its cues predict it.

206. **Before:** For example, a film with 40 cues gets Action if at least 21  
   **After:** For example, a film with 40 cues is assigned Action if at least 21

207. **Before:** It beats direct VGGish (0.403) by  
   **After:** It outperforms direct VGGish (0.403) by

208. **Before:** The emotion pipeline finds more of the true genres (0.651 against  
   **After:** The emotion pipeline recovers more of the true genres (0.651 against

209. **Before:** The emotion pipeline hardly moves (0.508 to 0.496),  
   **After:** The emotion pipeline changes little (0.508 to 0.496),

210. **Before:** Two further designs complete the picture  
   **After:** Two further designs complete the analysis

211. **Before:** and the held-out Eerola films touch neither fit.  
   **After:** and the held-out Eerola films enter neither fit.

212. **Before:** With both datasets in training, the lead is gone. Per film, the emotion pipeline (0.531) and direct VGGish (0.543) are level  
   **After:** With both datasets in training, the lead disappears. Per film, the emotion pipeline (0.531) and direct VGGish (0.543) are on par

213. **Before:** Scores from a single split move by a few hundredths with such choices.  
   **After:** Scores from a single split change by a few hundredths with such choices.

214. **Before:** The paragraphs after it sum up the results along the pipeline.  
   **After:** The paragraphs after it summarise the results along the pipeline.

215. **Before:** With per-genre thresholds the pipelines come together.  
   **After:** With per-genre thresholds the pipelines converge.

216. **Before:** the emotion pipeline beats direct VGGish at the default threshold  
   **After:** the emotion pipeline outperforms direct VGGish at the default threshold

217. **Before:** and with both datasets in training the pipelines are level.  
   **After:** and with both datasets in training the pipelines are on par.

218. **Before:** Under four further master seeds the mean of an entry moves by at most 0.047,  
   **After:** Under four further master seeds the mean of an entry changes by at most 0.047,

219. **Before:** In the second implementation the PCA-8 control moves from 0.417 to 0.457,  
   **After:** In the second implementation the PCA-8 control rises from 0.417 to 0.457,

## Chapter 6, Discussion (50 changes)

220. **Before:** Section~\ref{sec:disc_main} asks why the emotion step helps in some settings and not in others, and Section~\ref{sec:disc_bottleneck} looks at what the emotions add beyond making the input small.  
   **After:** Section~\ref{sec:disc_main} examines why the emotion step helps in some settings and not in others, and Section~\ref{sec:disc_bottleneck} considers what the emotions contribute beyond reducing the size of the input.

221. **Before:** Well above chance, but far from perfectly.  
   **After:** The emotions predict genre well above chance, but far from perfectly.

222. **Before:** The distance to the chance floors overstates the skill, however.  
   **After:** The distance to the chance floors overstates the predictive ability, however.

223. **Before:** The clearer evidence is the ranking:  
   **After:** The stronger evidence is the ranking:

224. **Before:** where that classifier cannot order any.  
   **After:** whereas that classifier cannot order any.

225. **Before:** although a better emotion model might still gain a little.  
   **After:** although a better emotion model might still yield a small improvement.

226. **Before:** Noise in the ratings is therefore not what limits the regressor.  
   **After:** Noise in the ratings is therefore not the main limitation of the regressor.

227. **Before:** Most genres have an emotional fingerprint that agrees with  
   **After:** Most genres have an emotional signature that is consistent with

228. **Before:** The signatures hold on the Blockbuster films ($r=0.836$), so the structure of the link between emotion and genre goes beyond the dataset it was found on.  
   **After:** The signatures hold on the Blockbuster films ($r=0.836$), so the structure of the link between emotion and genre extends beyond the dataset in which it was found.

229. **Before:** The answer is twofold.  
   **After:** The answer has two parts.

230. **Before:** How large its lead is over the strongest direct representations, however, cannot be settled on 40 films.  
   **After:** The size of its lead over the strongest direct representations, however, cannot be determined reliably on 40 films.

231. **Before:** It belongs to the emotion step, since it appears with each  
   **After:** It is a property of the emotion step, since it appears with each

232. **Before:** gives a genre result level with VGGish once its output passes through the emotions.  
   **After:** gives a genre result on par with VGGish once its output passes through the emotions.

233. **Before:** Most of that lead in $F_1$ comes from where the threshold lies, though.  
   **After:** Most of that lead in $F_1$, however, depends on the position of the threshold.

234. **Before:** What survives is the ranking of films,  
   **After:** What remains is the ranking of films,

235. **Before:** and with both datasets in training the pipelines are level.  
   **After:** and with both datasets in training the pipelines are on par.

236. **Before:** Table~\ref{tab:disc_settings} shows a clear pattern.  
   **After:** Table~\ref{tab:disc_settings} shows a consistent pattern.

237. **Before:** It is level with direct audio where the classifier was trained  
   **After:** It is on par with direct audio where the classifier was trained

238. **Before:** The pooled design shows both sides at once.  
   **After:** The pooled design shows both effects at once.

239. **Before:** This is a narrower claim than ``emotion predicts genre better than audio'', and it is the one the results support.  
   **After:** This is a narrower claim than ``emotion predicts genre better than audio'', and it is the claim the results support.

240. **Before:** Three explanations support each other.  
   **After:** Three explanations are mutually consistent.

241. **Before:** This fits the pattern above, where direct audio catches up exactly where 110 training films are available. It also fits the comparison over representations.  
   **After:** This is consistent with the pattern above, where direct audio reaches the same level exactly where 110 training films are available. It is also consistent with the comparison over representations.

242. **Before:** This is useless when the evaluation is grouped by film.  
   **After:** This information is of no use when the evaluation is grouped by film.

243. **Before:** and its handicap is gone.  
   **After:** and its disadvantage disappears.

244. **Before:** The emotion pipeline wins mainly by recall on the rare genres (Section~\ref{sec:eval_pergenre}). It finds them, at the price of more false positives. With a threshold per genre tuned on the training folds the pipelines come together in-domain,  
   **After:** The emotion pipeline leads mainly through recall on the rare genres (Section~\ref{sec:eval_pergenre}). It identifies them, at the cost of more false positives. With a threshold per genre tuned on the training folds the pipelines converge in-domain,

245. **Before:** so a small shift of the threshold changes its decisions a lot  
   **After:** so a small shift of the threshold changes its decisions considerably

246. **Before:** Tuning per genre needs labelled data, and it becomes unreliable for genres with only a handful of films. On a new dataset, too, the emotion pipeline barely changes when the threshold is tuned, while direct audio needs the tuning to catch up. Still, this advantage is smaller  
   **After:** Tuning per genre requires labelled data, and it becomes unreliable for genres with only a few films. On a new dataset, too, the emotion pipeline changes little when the threshold is tuned, while direct audio requires the tuning to reach a similar level. Nevertheless, this advantage is smaller

247. **Before:** A shift of the threshold alone cannot do that,  
   **After:** A shift of the threshold alone cannot produce this,

248. **Before:** A sceptic might say that eleven numbers simply overfit less than 128 on a dataset of about 330 clips.  
   **After:** One possible objection is that eleven features simply overfit less than 128 on a dataset of about 330 clips.

249. **Before:** The emotion pipeline still stays ahead of the control under every seed. Under transfer the picture is clearer in the original zero-shot run.  
   **After:** The emotion pipeline nevertheless stays ahead of the control under every seed. Under transfer the evidence is clearer in the original zero-shot run.

250. **Before:** The strongest evidence that the emotions mean more than their number is the replication of the signatures.  
   **After:** The strongest evidence that the emotions contribute more than a small number of dimensions is the replication of the signatures.

251. **Before:** This is one reason why richer representations cannot get more out of the music with the data available.  
   **After:** This is one reason why richer representations cannot extract more genre information from the music with the data available.

252. **Before:** because the discrete categories make the model easy to read.  
   **After:** because the discrete categories make the model easier to interpret.

253. **Before:** A better emotion model might therefore help a little, but by much less than its gap in $R^2$ suggests. A more accurate regressor did not reliably help.  
   **After:** A better emotion model might therefore help slightly, but by much less than its gap in $R^2$ suggests. A more accurate regressor did not reliably improve the genre result.

254. **Before:** \subsection{What the bottleneck buys in readability}  
   **After:** \subsection{Interpretability of the bottleneck}

255. **Before:** Apart from accuracy, the emotion pipeline gives what it was designed for. Each genre the model predicts well has an emotional signature that can be read.  
   **After:** Apart from accuracy, the emotion pipeline provides what it was designed for. Each genre the model predicts well has an interpretable emotional signature.

256. **Before:** is predicted poorly for how often it occurs.  
   **After:** is predicted poorly relative to its frequency.

257. **Before:** Its signature is the strongest, yet the emotion pipeline finds only half of the Horror films  
   **After:** Its signature is the strongest, yet the emotion pipeline identifies only half of the Horror films

258. **Before:** The random forest pulls the extreme fear of the Horror clips towards the average.  
   **After:** The random forest shifts the extreme fear of the Horror clips towards the average.

259. **Before:** Their film-level $F_1$ therefore moves a lot when a single film changes,  
   **After:** Their film-level $F_1$ therefore changes considerably when a single film changes,

260. **Before:** Only more films could settle them.  
   **After:** Only more films could resolve them.

261. **Before:** At that threshold, a classifier that assigns every genre to every film is also hard to beat.  
   **After:** At that threshold, a classifier that assigns every genre to every film is also difficult to outperform.

262. **Before:** On 40 films, macro-$F_1$ depends heavily on how often a pipeline says yes,  
   **After:** On 40 films, macro-$F_1$ depends heavily on how often a pipeline assigns a genre,

263. **Before:** The comparison with \citet{ma2021computational} is like for like.  
   **After:** The comparison with \citet{ma2021computational} is direct, since it uses their data and protocol.

264. **Before:** Under repeated cross-validation the learned embedding does not beat their full set of hand-crafted features. This agrees with their own careful wording. It also agrees with  
   **After:** Under repeated cross-validation the learned embedding does not outperform their full set of hand-crafted features. This is consistent with their own cautious wording. It is also consistent with

265. **Before:** Here the same happens to genre models moved from one film dataset to the next.  
   **After:** Here the same applies to genre models transferred from one film dataset to another.

266. **Before:** The contribution of this thesis is therefore not a higher number.  
   **After:** The contribution of this thesis is therefore not a higher score.

267. **Before:** On these data, a classifier that always predicts the most frequent genres wins on it outright.  
   **After:** On these data, a classifier that always predicts the most frequent genres achieves the best value on it.

268. **Before:** Horror therefore looks like a tense action or crime film  
   **After:** Horror therefore resembles a tense action or crime film

269. **Before:** Its point estimates move by a few hundredths with legitimate choices  
   **After:** Its point estimates change by a few hundredths with legitimate choices

## Appendix A, Supplementary Results (46 changes)

270. **Before:** The metrics in the last block reward predicting the frequent genres and hardly care about the rare ones.  
   **After:** The metrics in the last block reward predicting the frequent genres and give little weight to the rare ones.

271. **Before:** which predicts Drama for every clip, beats every model.  
   **After:** which predicts Drama for every clip, outperforms every trained model.

272. **Before:** A metric under which a model that has learned nothing ranks first is not a good headline for a question about genres such as Horror and Comedy.  
   **After:** A metric under which a model that has learned nothing ranks first is not a suitable primary measure for a question about genres such as Horror and Comedy.

273. **Before:** The all-positive classifier fails in the opposite way. Saying yes to every genre gives it a recall of 1.  
   **After:** The all-positive classifier fails in the opposite way. Assigning every genre gives it a recall of 1.

274. **Before:** Hamming loss and exact match expose it (0.691 and 0.000), and so do the threshold-free metrics.  
   **After:** Hamming loss and exact match reveal its weakness (0.691 and 0.000), and so do the threshold-free metrics.

275. **Before:** No metric that needs a threshold catches both trivial classifiers. ROC-AUC catches both,  
   **After:** No metric that requires a threshold identifies both trivial classifiers. ROC-AUC identifies both,

276. **Before:** Hamming loss deserves a closer look.  
   **After:** Hamming loss requires closer examination.

277. **Before:** and it even falls behind the random guess (0.323).  
   **After:** and it is even worse than the random guess (0.323).

278. **Before:** A needless positive therefore costs as much as a missed genre.  
   **After:** An unnecessary positive prediction therefore costs as much as a missed genre.

279. **Before:** All of the difference is in Comedy  
   **After:** The entire difference lies in Comedy

280. **Before:** This is the operating point driven by recall, described below, seen from the other side. Macro-$F_1$ rewards finding a rare genre; Hamming loss punishes every attempt.  
   **After:** This is the recall-driven operating point described below, viewed from the other side. Macro-$F_1$ rewards identifying a rare genre; Hamming loss penalises every attempt.

281. **Before:** The effect therefore belongs to the emotion features themselves,  
   **After:** The effect is therefore a property of the emotion features themselves,

282. **Before:** The decision threshold matters more.  
   **After:** The decision threshold has a larger effect.

283. **Before:** Over clips, the four pipelines then come together at between 0.420 and 0.433,  
   **After:** Over clips, the four pipelines then converge at between 0.420 and 0.433,

284. **Before:** Moving it does not buy precision.  
   **After:** Moving it does not increase precision.

285. **Before:** films and scores 0.435 to 0.445, like the classifier that says yes to everything.  
   **After:** films and scores 0.435 to 0.445, like the all-positive classifier.

286. **Before:** Over the seeds, the ten-repeat mean moves by 0.005 to 0.019.  
   **After:** Over the seeds, the ten-repeat mean changes by 0.005 to 0.019.

287. **Before:** What uncertainty is left is due to the small number of films.  
   **After:** The remaining uncertainty is due to the small number of films.

288. **Before:** Smaller subsets lose a little, but not significantly.  
   **After:** Smaller subsets perform slightly worse, but not significantly.

289. **Before:** The eight ratings move together: when tension goes up, fear tends to go up and valence down. In effect the genre signal runs along a single line, from threat to warmth. Fear is the most telling emotion,  
   **After:** The eight ratings are correlated: when tension increases, fear tends to increase and valence to decrease. In effect, the genre signal lies along a single axis, from threat to warmth. Fear is the most informative emotion,

290. **Before:** and all eight are kept because they make the model easier to read.  
   **After:** and all eight are kept because they make the model easier to interpret.

291. **Before:** That is level with the VGGish pipeline and not significantly below the ground-truth ceiling  
   **After:** This is on par with the VGGish pipeline and not significantly below the ground-truth ceiling

292. **Before:** A non-linear model alone does not help either:  
   **After:** A non-linear model alone does not improve the result either:

293. **Before:** Used directly, wav2vec~2.0 never predicts Comedy and hardly ever Horror  
   **After:** Used directly, wav2vec~2.0 never predicts Comedy and rarely Horror

294. **Before:** A linear classifier cannot find it from about 32 training films. The emotion ratings, in contrast, tell the random forest which acoustic properties matter.  
   **After:** A linear classifier cannot extract it from about 32 training films. The emotion ratings, in contrast, indicate to the random forest which acoustic properties are relevant.

295. **Before:** The rarest genres are almost never caught.  
   **After:** The rarest genres are almost never identified.

296. **Before:** is predicted poorly for how often it occurs (0.38 against 0.39 for Comedy with four films). This fits its missing emotional signature  
   **After:** is predicted poorly relative to its frequency (0.38 against 0.39 for Comedy with four films). This is consistent with its missing emotional signature

297. **Before:** the learned embedding (0.593) does not beat the full set  
   **After:** the learned embedding (0.593) does not outperform the full set

298. **Before:** Predicting the emotions of each cue and averaging them afterwards beats predicting them once  
   **After:** Predicting the emotions of each cue and averaging them afterwards outperforms predicting them once

299. **Before:** This is regression to the mean.  
   **After:** This is a regression towards the mean.

300. **Before:** The ratings show that the emotional signal itself is there.  
   **After:** The ratings show that the emotional signal itself is present.

301. **Before:** This leads it to predict too many genres.  
   **After:** As a consequence, it predicts too many genres.

302. **Before:** Finally, grouping by film matters.  
   **After:** Finally, grouping by film is essential.

303. **Before:** The emotion pipeline beats this reference significantly only on Comedy. It is level on Horror and behind on Crime and Drama, though not significantly.  
   **After:** The emotion pipeline outperforms this reference significantly only on Comedy. It is on par on Horror and behind on Crime and Drama, though not significantly.

304. **Before:** Nothing beats always predicting Drama at film level.  
   **After:** No pipeline outperforms always predicting Drama at film level.

305. **Before:** The reason is how often each pipeline says yes.  
   **After:** The reason is how often each pipeline assigns a genre.

306. **Before:** The emotion pipeline hardly changes with tuning on clips,  
   **After:** The emotion pipeline changes little with tuning on clips,

307. **Before:** because the default threshold suits them badly on the new dataset.  
   **After:** because the default threshold is poorly suited to them on the new dataset.

308. **Before:** The tuned scores therefore end close together,  
   **After:** The tuned scores therefore end close to each other,

309. **Before:** Most of the lead in $F_1$ at the default threshold thus comes from where the cut falls,  
   **After:** Most of the lead in $F_1$ at the default threshold thus depends on the position of the cut,

310. **Before:** The real picture differs from the schematic in three ways.  
   **After:** The empirical structure differs from the schematic in three ways.

311. **Before:** which fits the finding that one to three emotions keep most of that signal.  
   **After:** which is consistent with the finding that one to three emotions retain most of that signal.

312. **Before:** Strong regularisation pulls the probabilities of a logistic regression towards one half,  
   **After:** Strong regularisation shifts the probabilities of a logistic regression towards one half,

313. **Before:** the mean of an emotion pipeline moves by at most 0.042  
   **After:** the mean of an emotion pipeline changes by at most 0.042

314. **Before:** That of a direct pipeline moves by at most 0.047 (VGGish).  
   **After:** That of a direct pipeline changes by at most 0.047 (VGGish).

315. **Before:** so that predicted Horror looks like a tense, energetic action or crime film.  
   **After:** so that predicted Horror resembles a tense, energetic action or crime film.
