# Actuarial loss prediction

This project work was prepared for the closure of the [Mathematics Expert in Data Analytics and Machine Learning](https://ai.elte.hu/) postgraduate specialization program first offered in 2021/22 by the Institute of Mathematics of Eötvös Loránd University.

## Origin
The source of the dataset and the problem is a Kaggle competition entitled “[Actuarial loss prediction](https://www.kaggle.com/c/actuarial-loss-estimation/)”. The competition was held in 2020/21 and was hosted by [The Actuaries Institute of Australia](https://www.actuaries.asn.au/), [Institute and Faculty of Actuaries](https://actuaries.org.uk/) and the [Singapore Actuarial Society](https://www.actuaries.org.sg/). The challenge was to predict the ultimate claims for a synthetic portfolio of workers compensation claims. The contestants were provided a training set where the ultimate claim (target variable) is known and a test set where the target had to be predicted and the real values are unknown. The evaluation was based on the RMSE on the test set, 
$$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(y_i-\widetilde{y_i}\right)^2}$$
where $n$ denotes the number of observations, $y_i$ the true and $\widetilde{y_i}$ the predicted value of the $i$-th observation.

## Insurance terms

Before introducing the dataset and its features, I will briefly explain some basic terms relating to insurance. An **insurance claim** is a formal request by a policyholder to an insurance company for coverage or compensation for a covered loss or policy event. [Workers' compensation insurance](https://en.wikipedia.org/wiki/Workers%27_compensation) “is a form of insurance providing wage replacement and medical benefits to employees injured in the course of employment in exchange for mandatory relinquishment of the employee's right to sue his or her employer for the tort of negligence" . The **insurance claim process** contains all the steps that is needed for an insurance claim to be closed. This include for example, but not exclusively the following:

1. Claim occurrence
2. Reporting the claim to the insurer
3. Claim investigation and initial claim estimation
4. Processing, iteration
5. Indemnity (an insurance claim can be closed with full, partial or nil indemnity)

The **incurred date** of an insurance claim is the date (and time) when the insured event occurred. The **reporting date** of an insurance claim is the date (and time) when the insured event is reported directly to the insurer, so there can be a **delay in the reporting** of an insurance claim. It is not unusual that the insurer is not aware of already incurred claims, although workers’ compensation claims are expected to be reported swiftly (in weeks). After a claim is reported a claim investigation expert inspects the individual claim and assesses whether the conditions of the insurance contract are fulfilled. If so, then a certain amount is set aside as **provision** to cover the insurer’s **future obligations**. This amount is not paid up-front, there can be iterations in the process: for example it can turn out that the injury of a person is worse than the initial assessment, so the ultimate claim amount can change; there are usually costs and expenses as well that relate to the claim process (e.g. administrative expenses, fee of the claim investigation experts, legal costs). We note that the **ultimate incurred claim cost** can be different from what the **initial incurred claim cost** was. Our goal in this theses is to predict the ultimate incurred claim cost. Insurance claims usually have heavy-tailed, skewed distribution.

## Dataset
After the brief introduction if insurance terminology I will continue with describing the dataset. “The dataset provided includes 90,000 realistic, synthetically generated worker compensation insurance policies, all of which have had an accident. For each record there is demographic and worker related information, as well as a text description of the accident.”  The training set contains 54,000 records where the target is known, the remainder is referred to as the test set where the target is unknown.

The dataset contains the following features (the target variable in the last row):

| Feature name  | Data type | Description |
| --- | --- | --- |
| `ClaimNumber`| integer | Unique policy identifier
| `DateTimeOfAccident`| datetime | Date and time of accident
| `DateReported`| date | Date that accident was reported
| `Age`| integer | Age of worker
| `Gender`| categorical | Gender of worker. (M)ale, (F)emale, (U)nknown.
| `MaritalStatus`| categorical | Martial status of worker. (M)arried, (S)ingle, (U)nknown.
| `DependentChildren`| integer | The number of dependent children
| `DependentsOther`| integer | The number of dependants excluding children
| `WeeklyWages`| float | Total weekly wage
| `PartTimeFullTime`| categorical | Binary (P) or (F)
| `HoursWorkedPerWeek`| integer | Total hours worked per week
| `DaysWorkedPerWeek`| integer | Number of days worked per week
| `ClaimDescription`| string | Free text description of the claim
| `InitialIncurredClaimCost`| float | Initial estimate by the insurer of the claim cost
| `UltimateIncurredClaimCost`| float | Total claims payments by the insurance company. This is the field you are asked to predict in the test set.

Please find below a randomly selected, 4-element sample of the dataset (the ultimate claims are rounded to the nearest integer).

| ClaimNumber	| WC9460849	| WC9764859 |	WC2330788 |	WC3015328 |
| --- | --- | --- | --- | --- |
| DateTimeOfAccident	|2004. 9. 10. 0:00 |	2005. 1. 27. 14:00	 |	1989. 8. 30. 12:00	 |	1991. 7. 29. 8:00|
| DateReported|	2005.01.20 |	2005.02.23 |	1989.09.09 |	1991.08.15|
| Age|	58 |		23 |		24 |		20|
| Gender	|F	 |	F	 |	M |		M|
| MaritalStatus|	M	 |	S	 |	M |		S|
| DependentChildren|	0 |		0	 |	0 |		0|
| DependentsOther	|1	0	 |	0	 |	0|
| WeeklyWages|	866	 |	500	 |	310	 |	232.18|
| PartTimeFullTime|	F |		F |		F |		F|
| HoursWorkedPerWeek|	36	 |	40	 |	40	 |	38|
| DaysWorkedPerWeek|	5 |		5	 |	5	 |	5|
| ClaimDescription|	UNLOADING BOXES BACK	 |	STRUCK METAL STRAPPING LACERATION RING FINGER |		CUT ON STEEL PLATFORM LEFT SHOULDER STRAINED MUSCLE SHOULDER |		CUT ON MEAT HOOK CUT FINGER CUT|
| InitialIncurredClaimsCost	|96,807	 |	1,000 |		650	 |	300|
| UltimateIncurredClaimCost|	164,893	 |	903	 |	814 |		359|

## My approach

For this project work I turn my attention to a certain variable in the dataset which I previously neglected, namely the column `ClaimDescription` which contains a description of the claims in English. I will use this feature and apply natural language processing techniques to prepare and clean the data and use `word based vectorisation`, `embedding` and `LSTM` which is an artificial recurrent neural network model to predict the target variable that is the ultimate incurred claims.

Exploratory data analysis is a crucial task in every machine learning problem. For this reason first I will summarise the main findings of the data analysis and describe the decisions that I used to engineer new features. I will discuss in more depths the two main features in my scope, namely the `ClaimDescription` and the `UltimateIncurredClaimCost`. The thesis has a limited scope in modelling, but the data analysis section can provide ideas and insights to further extend the modelling scope in the future.

In the next step I will grasp the modelling problem in two ways. On the one hand, as a supervised classification problem and on the other hand as a supervised regression problem. In the former I will create labels on the dataset using the original target feature, i.e. bin the observations annually based on the quartiles. In the latter case I will model the numerical value of the log-transformed target.

Finally, I will evaluate the models and the lessons learnt and highlight possible extensions for the future.

Before diving deep into the discussion I would like highlight certain limitations of the chosen scope and modelling approaches and elaborate what my expectations are on the results with the knowledge of these limitations in mind.
 - I am excluding most of the features from the modelling and only using the textual description of claims when predicting the target. By doing this information is inherently lost, hence the predictive power of our model will be limited.
 -Even if we included all our predictors and other engineered features we may have not sufficient information on the run-off of the claims in this specific workers compensation insurance portfolio. I would like to emphasize the fact that we have no information whatsoever on what happens between the initial recognition of a claim and the event when the claim is finally settled. This would be a crucial information to better understand the portfolio and by using this knowledge to provide better results. For example, there may be many follow-up medical examinations where a certain injury may turn out to have more severe, metastatic consequences thus would have a higher final claim amount than what was primarily recognised. Another example would be juridical debates that the insured worker and the insurance company may have for example on deciding what kind of medical treatments and medication should be covered by the insurance company. A final example would be the fact that such dataset is usually created on a case-by-case basis, manually, by the claim investigator, consequently it may be prone to man-made errors. Our example is a synthetic dataset, but it is claimed that it reflects real word portfolios closely, so we could assume that this is handled in a way that the synthetic dataset contains a lot of noise which is designed to mimic errors in data recording.
 - The designated metric for the original competition is the root means squared error. This metric punishes larger errors heavily. In the case when the problem is tackled as a regression problem this will mean that for high severity claims (i.e. claims with the highest amount) mistakes in the prediction will have a more severe effect on the metric.
 - The target variable has its own setbacks. The distribution of insurance claims usually have high frequency for observations with low severity, but there is a part of the distribution which behaves differently and should not be neglected. These are the cases that are usually outliers of the distribution but they should not be simply dropped because such claims are not uncommon. The general practice is that these claims are handled individually on a case-by-case basis which in this problem I am unable to perform. For this reason I am building a model that tries to generalise this double sided behaviour, so I would expect that this modelling approach would lead to errors in the prediction as well.
 - We lack insurance portfolio and market characteristic knowledge. We will see in the data analysis part that there is a visible inflation of claims. We will also see that there are years where the gap between the initial and ultimate claim increased significantly, then decreased again. I analysed these observations with the information that was available. I also tried to explain these movements in light of external indicators such as consumer price index, gross domestic product and exchange rates. These analyses were based on heavy assumptions about the origin of the data (which is unknown) and all efforts trying to find an explanation on the movements seemed to be in vain. Without portfolio and market knowledge it is hard to create a generalised model. For example, our model may generalise on patterns that were observable in the past, but should not extrapolate on future predictions. There may be changes in the claim settling process of the insurance company, or there may be legislation changes, or it is possible that the workers compensation portfolio was inherited from an insurance company to another insurer. These can be all valid contributors to the target on which we have no information, so our model’s predictive power will be limited.


