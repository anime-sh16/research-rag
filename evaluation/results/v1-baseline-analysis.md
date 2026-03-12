## Overall Analysis
problems int he currnet version include:
- poor chinking of fixed length
- loss of tabular and visual data, especially tabular.
- retrieval problem: Almost all similarity score aore arounf 0.7 value
- focusing onone of the two souces more.
- overall highly faithfull. always have high value when (about)correct answer. when wrong or `dont know`, sometimes high, when it explain why.


## Analysis based on question type
1. cross-paper:
- Sample 1:
  * spread_diversity= 5;  retireval_score_spread = 0.03
  * Actual source missing, but similar sources' chunks used.
  * low values: Context recall, and answer relevance. context precision reasonalble, not very good.
- Sample 2"
  * spread_diversity= 2;  retireval_score_spread = 0.03
  * Only 1 source correct which were 4 out of 5 chunks. other source incorrect.
  * context recall and precision very low
  * generated `I don't have enough information to answer this`
- Sample 3:
  * spread_diversity= 4;  retireval_score_spread = 0.02
  * out of 5, only 1 chuck is from correct source. other 1 is a similar topic to source. rest 3 are general surveys,
  * All 4 metrics very low
  * generated `I don't have enough information to answer this`
- Sample 4:
  * spread_diversity= 1;  retireval_score_spread = 0.008
  * All 5 chunks are from singel source, correct one, but missing the second source.
  * context recall and precision very low
  * generated `I don't have enough information to answer this`
- Sample 5:
  * spread_diversity= 1;  retireval_score_spread = 0.05
  * All 5 chunks are from singel source, correct one, but missing the second source.
  * anser relevancy and context recall very low. faithfulness zero as not answer produced.
  * generated `I don't have enough information to answer this`
- Sample 6:
  * spread_diversity= 2;  retireval_score_spread = 0.025
  * Only 1 source correct which were 4 out of 5 chunks. other source incorrect.
  * context recall and precision very low
  * generated `I don't have enough information to answer this`

2. Factual
- Most of teh samples perform very well.
- 2 samples have low context precision ut correct source

3. Conceptual
- All samples have high metric value.
- one sample generated `I don't have enough information to answer this`
  - in this, teh context had 1 chunk from wring source.
  - but since After saying that the model doesnt know the answer, it explain that even thought he data is present the exact ansswer to the question is not....

4. multi-hop
- 4 out of 6 samples have good result. other two have very low answer relevance, contect recall and precision.
- out of 6 samples, 4 have more than one sources which is wrong as all multi-hop were single source.
- despite all this, all 6 samples have very high faithfullness.

5. numerical
- Sample 1:
  * very low answer relevance, context recall and precision.
  * 4 out of 5 chucks from correct source.
  * generated `I don't have enough information to answer this`
  * answer lost due to being in table.
- Sample 2:
  * very low context precision
  * all chunks fro correct paper
- Sample 3:
  * low context precision
  * all chunks fro correct paper
- Sample 4 and 5:
  * decently good reslut.
