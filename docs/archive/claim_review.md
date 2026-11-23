# Strong claims softened: every changed phrase

After the supervisor's comment of 12 Nov 2026 (avoid strong claims, and words such as "find out" or "determine" that present an interpretation as a discovery or as certain), the thesis was revised throughout. Measured results are still stated plainly; what was changed are interpretations, causal explanations and generalisations, which now say "suggests", "appears", "may", "is consistent with" or "in these data". Words such as "clearly", "the best", "defined by", "confirms", "explains why", "is ruled out" and "determine" were replaced where they claimed more than a test supports. Three headings changed in Chapter 5: "Replication of the Signatures" (was a question), "Reverse and Pooled Transfer" (was "Transfer in Every Direction") and "Summary of Results" (was "Summary of Findings"); labels unchanged. A script checked that every number, citation, reference and label is unchanged in each file. Each pair is *before* / *after* in the LaTeX source; say which to undo (quote the number).

## Chapter 1: Introduction (6 changes)

1. **Before:** but they cannot explain why a soundtrack is assigned a particular genre.  
   **After:** but they offer no direct explanation of why a soundtrack is assigned a particular genre.

2. **Before:** At the same time, music research shows that emotion and genre  
   **After:** At the same time, music research reports that emotion and genre

3. **Before:** First, a prediction that rests on emotions can be explained and checked.  
   **After:** First, a prediction that rests on emotions can, in principle, be explained and checked.

4. **Before:** The aim is to determine how well soundtrack emotion  
   **After:** The aim is to examine how well soundtrack emotion

5. **Before:** This tests whether the emotional step  
   **After:** This examines whether the emotional step

6. **Before:** reports the checks of the main results, none of which changes them.  
   **After:** reports further checks of the main results.

## Chapter 2: Fundamentals (5 changes)

7. **Before:** This has a direct consequence for the method.  
   **After:** This motivates the method.

8. **Before:** and never asks why audio should predict genre at all. This thesis treats emotion as the reason for that link instead.  
   **After:** and does not ask why audio should predict genre at all. This thesis instead treats emotion as a possible reason for that link.

9. **Before:** and a model should be able to recover and use them.  
   **After:** and a model may be able to recover and use them.

10. **Before:** Whether it does is what the evaluation has to establish,  
   **After:** Whether it does is what the evaluation has to test,

11. **Before:** proved far too weak for every feature set  
   **After:** turned out to be far too weak for every feature set

## Chapter 3: Related Work (5 changes)

12. **Before:** It is an established finding in music information retrieval, it has been replicated by independent groups, and it has been studied from both directions.  
   **After:** It is a well-documented finding in music information retrieval, reported by independent groups and studied from both directions.

13. **Before:** Together, these studies show that the link between emotion and genre holds in both directions and is replicable. They also show that it depends  
   **After:** Together, these studies suggest that the link between emotion and genre holds in both directions and can be replicated. They also indicate that it depends

14. **Before:** The choice of audio representation alone therefore explains little about genre.  
   **After:** This suggests that the choice of audio representation alone explains little about genre.

15. **Before:** The weakness of raw audio is, in fact, the subject  
   **After:** The weakness of raw audio is the subject

16. **Before:** It determines automatically which tracks  
   **After:** It identifies automatically which tracks

## Chapter 4: Methods (13 changes)

17. **Before:** These five are established in the literature and have  
   **After:** These five are common in the literature and have

18. **Before:** and their first ten seconds already describe them well.  
   **After:** and their first ten seconds appear to describe them well enough for these tasks.

19. **Before:** On the spectrogram embeddings the three regressors are in fact close  
   **After:** On the spectrogram embeddings the three regressors are close

20. **Before:** and is the most robust choice across inputs.  
   **After:** and gave the most consistent results across inputs.

21. **Before:** it is clearly best.  
   **After:** it scores highest.

22. **Before:** but the out-of-fold procedure is the correct one, and it is the one reported.  
   **After:** but the out-of-fold procedure is the methodologically appropriate one, and it is the one reported.

23. **Before:** The linear model is also more useful here, because  
   **After:** The linear model is also easier to interpret here, because

24. **Before:** so a positive result cannot be attributed to a weak baseline.  
   **After:** so a positive result is less likely to be due to a weak baseline.

25. **Before:** It marks the ceiling of the emotion pipeline:  
   **After:** It approximates the ceiling of the emotion pipeline:

26. **Before:** The gap between this arm and the arm with predicted emotions is the cost of predicting the emotions.  
   **After:** The gap between this arm and the arm with predicted emotions estimates the cost of predicting the emotions.

27. **Before:** the PCA-8 control would outperform it as well.  
   **After:** the PCA-8 control would be expected to outperform it as well.

28. **Before:** To determine which emotions are associated with which genres  
   **After:** To examine which emotions are associated with which genres

29. **Before:** A score well above it confirms that the cached embeddings  
   **After:** A score well above it indicates that the cached embeddings

## Chapter 5: Evaluation (24 changes)

30. **Before:** Further checks that do not change these results are in Appendix  
   **After:** Further checks of these results are in Appendix

31. **Before:** descriptors are clearly weaker (0.476)  
   **After:** descriptors are weaker (0.476)

32. **Before:** are clearly the hardest.  
   **After:** are the hardest.

33. **Before:** and clearly above the base-rate random guess  
   **After:** and above the base-rate random guess

34. **Before:** ROC-AUC separates them clearly.  
   **After:** ROC-AUC separates them.

35. **Before:** Second, compression alone accounts for part of the gain at film level.  
   **After:** Second, compression alone may account for part of the gain at film level.

36. **Before:** These values are stable. Re-running  
   **After:** These values change little across seeds. Re-running

37. **Before:** A more accurate emotion model does not reliably yield a better genre result. Ridge  
   **After:** In these experiments, a more accurate emotion model did not reliably yield a better genre result. Ridge

38. **Before:** A post-hoc test, however, does not confirm this lead:  
   **After:** A post-hoc test, however, does not find this lead significant:

39. **Before:** The emotion pipeline is the best pipeline for Action, Crime and Comedy. For Drama the best is  
   **After:** The emotion pipeline scores highest for Action, Crime and Comedy. For Drama the highest score comes from

40. **Before:** Fear is the strongest single discriminator. It separates  
   **After:** In these data, fear is the strongest single discriminator. It separates

41. **Before:** {Do the Signatures Replicate?}  
   **After:** {Replication of the Signatures}

42. **Before:** to test whether the signatures describe film music in general or only the Eerola films.  
   **After:** to test whether the signatures extend beyond the Eerola films.

43. **Before:** The signatures replicate. Across all 48  
   **After:** The signatures largely replicate. Across all 48

44. **Before:** Horror is defined by fear in both datasets.  
   **After:** Horror is characterised by fear in both datasets.

45. **Before:** Genre information transfers between the datasets. The emotion pipeline reaches 0.508  
   **After:** Much of the genre information transfers between the datasets. The emotion pipeline reaches 0.508

46. **Before:** the emotion pipeline still ranks the Blockbuster films best, with a macro ROC-AUC of 0.763  
   **After:** the emotion pipeline still has the highest macro ROC-AUC on the Blockbuster films, 0.763

47. **Before:** {Transfer in Every Direction}  
   **After:** {Reverse and Pooled Transfer}

48. **Before:** Two further designs complete the analysis  
   **After:** Two further designs extend the analysis

49. **Before:** {Summary of Findings}  
   **After:** {Summary of Results}

50. **Before:** The emotion pipeline is the best pipeline for Action, Crime and Comedy, mainly  
   **After:** The emotion pipeline scores highest for Action, Crime and Comedy, mainly

51. **Before:** The signatures replicate on the Blockbuster dataset  
   **After:** The signatures largely replicate on the Blockbuster dataset

52. **Before:** although the emotion pipeline still ranks the films best.  
   **After:** although the emotion pipeline still has the higher ROC-AUC, not significantly.

53. **Before:** The box marks the best pipeline per genre;  
   **After:** The box marks the highest-scoring pipeline per genre;

## Chapter 6: Discussion (39 changes)

54. **Before:** examines why the emotion step helps in some settings and not in others  
   **After:** discusses possible reasons why the emotion step helps in some settings and not in others

55. **Before:** The stronger evidence is the ranking:  
   **After:** A more informative measure is the ranking:

56. **Before:** Emotions predicted from audio carry most of the genre information in the ratings they were trained on. The pipeline therefore needs no emotion labels once it is in use, although a better emotion model might still yield a small improvement.  
   **After:** Emotions predicted from audio appear to retain most of the genre information in the ratings they were trained on. Once trained, the pipeline needs no emotion labels, although a better emotion model might still yield a small improvement.

57. **Before:** Noise in the ratings is therefore not the main limitation of the regressor. The limit lies in the audio representations and the amount of training data.  
   **After:** This suggests that noise in the ratings is not the main limitation of the regressor, and that the limit lies more in the audio representations and the amount of training data.

58. **Before:** Horror is defined by fear, Comedy by the absence  
   **After:** Horror is characterised mainly by fear, Comedy by the absence

59. **Before:** Adventure has no signature, because it occurs together  
   **After:** Adventure has no signature, possibly because it occurs together

60. **Before:** The signatures hold on the Blockbuster films ($r=0.836$), so the structure of the link between emotion and genre extends beyond the dataset in which it was found.  
   **After:** The signatures largely hold on the Blockbuster films ($r=0.836$), which suggests that the structure of the link between emotion and genre extends beyond the dataset in which it was first observed.

61. **Before:** cannot be determined reliably on 40 films. The advantage does not depend on VGGish or on the random forest. It is a property of the emotion step, since it appears with each of six representations and three regressors  
   **After:** cannot be estimated reliably on 40 films. The advantage does not appear to depend on VGGish or on the random forest. It appears with each of six representations and three regressors, which points to the emotion step rather than to one particular model

62. **Before:** What the genre classifier needs from Stage~1 is therefore not the most accurate estimate per clip.  
   **After:** This suggests that the genre classifier does not depend mainly on the most accurate estimate per clip from Stage~1.

63. **Before:** Genre information transfers: trained on Eerola alone  
   **After:** Much of the genre information transfers: trained on Eerola alone

64. **Before:** where the emotion pipeline stays first.  
   **After:** where the emotion pipeline stays ahead, although not significantly.

65. **Before:** The advantage of the emotion pipeline is therefore an advantage when training data are scarce and when the dataset changes. This is a narrower claim than ``emotion predicts genre better than audio'', and it is the claim the results support.  
   **After:** The advantage of the emotion pipeline therefore appears to be an advantage when training data are scarce and when the dataset changes. This is a narrower claim than ``emotion predicts genre better than audio'', and it is the claim that these results support most directly.

66. **Before:** Three explanations are mutually consistent.  
   **After:** Three possible explanations are compatible with this pattern, and the experiments do not separate them.

67. **Before:** The emotion stage adds prior knowledge  
   **After:** The emotion stage may add prior knowledge

68. **Before:** This information is of no use when the evaluation is grouped by film.  
   **After:** This information is likely of little use when the evaluation is grouped by film.

69. **Before:** Emotion values are given on a rating scale that means the same in any dataset.  
   **After:** Emotion values are given on a rating scale that is intended to mean the same in any dataset.

70. **Before:** A model that works at the default threshold does have a practical advantage.  
   **After:** A model that performs well at the default threshold may still have a practical advantage.

71. **Before:** A shift of the threshold alone cannot produce this  
   **After:** A shift of the threshold alone would not produce this

72. **Before:** so compression explains part of the gain.  
   **After:** so compression may account for part of the gain.

73. **Before:** The strongest evidence that the emotions contribute more than a small number of dimensions is the replication of the signatures.  
   **After:** A further indication that the emotions contribute more than a small number of dimensions comes from the replication of the signatures.

74. **Before:** and Horror is defined by fear in both. A principal component has no meaning  
   **After:** and Horror is characterised by fear in both. A principal component has no predefined meaning

75. **Before:** shows that no single emotion is necessary.  
   **After:** suggests that no single emotion is necessary.

76. **Before:** The genre signal of the ratings lies along essentially one axis  
   **After:** The genre signal of the ratings appears to lie mainly along one axis

77. **Before:** This is one reason why richer representations cannot extract more genre information  
   **After:** This may be one reason why richer representations did not extract more genre information

78. **Before:** The same axis explains why the imperfect emotion predictions work almost as well as the ratings.  
   **After:** The same axis may also explain why the imperfect emotion predictions perform almost as well as the ratings.

79. **Before:** Apart from accuracy, the emotion pipeline provides what it was designed for. Each genre the model predicts well has an interpretable emotional signature.  
   **After:** Apart from accuracy, the emotion pipeline was designed to be interpretable, and the results suggest that it is, at least for the genres it predicts well. Each of these genres has an interpretable emotional signature.

80. **Before:** Horror shows how an error can be traced.  
   **After:** Horror illustrates how an error can be traced.

81. **Before:** The weakness comes from Stage~1, not from the genre classifier, which reaches 0.41 on Horror from the ratings.  
   **After:** The weakness appears to come mainly from Stage~1 rather than from the genre classifier, which reaches 0.41 on Horror from the ratings.

82. **Before:** Both the successes and the errors of the pipeline come with an explanation that can be checked against how film music is written.  
   **After:** For both the successes and the errors of the pipeline, a candidate explanation can be given and checked against how film music is written.

83. **Before:** At that point the analysis of alternative metrics showed that it fits the structure of the labels.  
   **After:** At that point the analysis of alternative metrics suggested that it fits the structure of the labels.

84. **Before:** Film-level scoring therefore does not favour the emotion pipeline.  
   **After:** Film-level scoring therefore does not appear to favour the emotion pipeline.

85. **Before:** audio reveals arousal more reliably  
   **After:** audio conveys arousal more reliably

86. **Before:** showed that an emotion model trained  
   **After:** reported that an emotion model trained

87. **Before:** What is new is that a model built on emotions loses less.  
   **After:** What this thesis adds is the observation that a model built on emotions lost less in this setting.

88. **Before:** found intense genres easier to separate by their scores than story-driven genres, and the signatures found here explain why. Genres with a conventional emotional identity can be recovered from the music, while genres defined by story or setting cannot.  
   **After:** reported that intense genres were easier to separate by their scores than story-driven genres, and the signatures observed here offer a possible explanation. Genres with a conventional emotional identity could be recovered from the music here, while genres defined mainly by story or setting could not.

89. **Before:** found audio to be the weakest of five kinds of input for genre, and the modest absolute scores reported here confirm it.  
   **After:** reported audio as the weakest of five kinds of input for genre, and the modest absolute scores reported here are consistent with that.

90. **Before:** It is a model whose predictions can be explained and which loses less when the data change. This answers the concern of  
   **After:** It is a model whose predictions can be related to emotions and which lost less when the data changed. This offers one response to the concern of

91. **Before:** used as a step in between, the emotion model becomes the explanation.  
   **After:** used as a step in between, the emotion model provides a partial explanation of the genre prediction.

92. **Before:** relates the findings to the literature.  
   **After:** relates the results to the literature.

## Chapter 7: Conclusions and Future Work (20 changes)

93. **Before:** The more reliable evidence is the ranking of films.  
   **After:** The ranking of films is more informative.

94. **Before:** so the pipeline needs no emotion labels once it has been trained.  
   **After:** so little appears to be lost by predicting the emotions instead of having listeners rate them.

95. **Before:** Fear is the most discriminating single emotion,  
   **After:** In these data, fear is the most discriminating single emotion,

96. **Before:** The signatures replicate on the Blockbuster films  
   **After:** The signatures largely replicate on the Blockbuster films

97. **Before:** so it is a property of the emotion step and not of one model.  
   **After:** which suggests that it stems from the emotion step rather than from one particular model.

98. **Before:** Compression alone accounts for part of the gain  
   **After:** Compression alone appears to account for part of the gain

99. **Before:** representations cannot be determined reliably.  
   **After:** representations cannot be estimated reliably.

100. **Before:** Genre information transfers between the datasets.  
   **After:** Much of the genre information transfers between the datasets.

101. **Before:** The emotion pipeline still ranks the films best (ROC-AUC 0.763 against 0.720)  
   **After:** The emotion pipeline still ranks the films slightly better (ROC-AUC 0.763 against 0.720)

102. **Before:** The emotion step does not make genre prediction from film music more accurate in general. Its advantage is specific to settings with little training data and to transfer between datasets  
   **After:** The results do not indicate that the emotion step makes genre prediction from film music more accurate in general. Its advantage appears to be limited to settings with little training data and to transfer between datasets

103. **Before:** Its main contribution is interpretability. Each prediction can be traced to emotions that listeners can name, and errors can be traced as well. The weakness on Horror, for example, comes from Stage~1, which shifts  
   **After:** Its main contribution lies in interpretability. Each prediction can be related to emotions that listeners can name, and some errors can be traced as well. The weakness on Horror, for example, can be traced mainly to Stage~1, which shifts

104. **Before:** which a direct audio model does not allow.  
   **After:** which is not possible in the same way for a direct audio model.

105. **Before:** The following directions follow directly from them.  
   **After:** The following directions follow from them.

106. **Before:** The main limitation is the number of films.  
   **After:** The most important limitation is probably the number of films.

107. **Before:** and genre labels is therefore the most important requirement. It would allow the advantage over the strongest direct representations to be measured with useful precision.  
   **After:** and genre labels would therefore be the most useful addition. It would allow the advantage over the strongest direct representations to be estimated with useful precision.

108. **Before:** was found after the results were known  
   **After:** was identified after the results were known

109. **Before:** which causes the weakness on Horror.  
   **After:** which appears to cause much of the weakness on Horror.

110. **Before:** would show how much of the remaining error  
   **After:** could show how much of the remaining error

111. **Before:** Such a study would show whether  
   **After:** Such a study could show whether

112. **Before:** Both are reported transparently, but further studies could avoid them.  
   **After:** Both are reported in this thesis, but further studies could avoid them.

## Appendix A (18 changes)

113. **Before:** The effect is therefore a property of the emotion features themselves, in which a tense crime clip resembles a horror clip. It does not come from errors of the emotion regression.  
   **After:** This suggests that the effect is a property of the emotion features themselves, in which a tense crime clip resembles a horror clip, rather than a result of errors of the emotion regression.

114. **Before:** in-domain advantage at the default threshold therefore comes from the operating point.  
   **After:** in-domain advantage at the default threshold therefore appears to come from the operating point.

115. **Before:** numbers are therefore stable against everything the code controls. The remaining uncertainty is due to the small number of films.  
   **After:** numbers are therefore stable against the random choices the code controls. The remaining uncertainty comes mainly from the small number of films.

116. **Before:** In effect, the genre signal lies along a single axis  
   **After:** In effect, the genre signal lies mainly along a single axis

117. **Before:** The benefit of the emotion step is therefore not tied to one embedding, and it is largest  
   **After:** The benefit of the emotion step therefore does not appear to be tied to one embedding, and it is largest

118. **Before:** Compression explains part of the gain over direct use.  
   **After:** Compression accounts for part of the gain over direct use.

119. **Before:** Only happiness, sadness and energy are clearly worse  
   **After:** Only happiness, sadness and energy are worse

120. **Before:** Its genre information is spread over 768 dimensions learnt from speech. A linear classifier cannot extract it from about 32 training films. The emotion ratings, in contrast, indicate to the random forest which acoustic properties are relevant. A shortcut through film identity is ruled out.  
   **After:** Its genre information may be spread over 768 dimensions learnt from speech, which a linear classifier may not be able to extract from about 32 training films. The emotion ratings, in contrast, may indicate to the random forest which acoustic properties are relevant. A shortcut through film identity appears unlikely.

121. **Before:** The ratings place Horror clearly above all other genres in fear.  
   **After:** The ratings place Horror well above all other genres in fear.

122. **Before:** This is a regression towards the mean.  
   **After:** This pattern is consistent with regression towards the mean.

123. **Before:** so it cannot reach the extreme values of a small group.  
   **After:** so it has difficulty reaching the extreme values of a small group.

124. **Before:** genre classifier therefore cannot separate Horror from these genres.  
   **After:** genre classifier therefore separates Horror poorly from these genres.

125. **Before:** would be the remedy.  
   **After:** may reduce this weakness.

126. **Before:** Performance is therefore limited by the amount of data  
   **After:** This suggests that performance is limited by the amount of data

127. **Before:** because the default threshold is poorly suited to them on the new dataset.  
   **After:** probably because the default threshold is poorly suited to them on the new dataset.

128. **Before:** pipeline ranks the films best, and its lead over the PCA-8 control is significant.  
   **After:** pipeline has the highest ROC-AUC, and its lead over the PCA-8 control is significant.

129. **Before:** The reason is how often each pipeline assigns a genre.  
   **After:** The main reason appears to be how often each pipeline assigns a genre.

130. **Before:** the finding that one to three emotions retain most of that signal.  
   **After:** the observation that one to three emotions retain most of that signal.
