# flip-or-skip

## Libraries and Development Environments
This project was developed entirely within Amazon SageMaker, using Jupyter Lab + Notebooks, S3, and CloudWatch. The following Python libraries are required for running the notebooks in their entirety: NumPy, Pandas, Sci-kit Learn, SageMaker, os, io, mxnet, and boto3. 

## Project Overview
The following content of this README is exclusively exerpts from the full report included in this repository, titled 'report.pdf'.
<br/>
This project focuses on real estate data, specifically past real estate sales and current real estate listings in Los Angeles County. Within the domain of real estate, this project will be considering how machine learning can be used to aid in identifying quality investment properties which can be purchased below market value, generally due to the property being in a distressed state, and then renovating the property in order for it to be sold at a profit. This process is commonly referred to as a “flip” within the domain of real estate.
<br/>
Finding properties which present a successful margin for a real estate “flip” can be challenging. The best indicator of a profitable real estate investment is a property which can be acquired below 70% of the value at which it could potentially be sold if it were in like-new condition, also referred to as “after repair value” (ARV)[1]. The total cost of the anticipated repairs needs to be accounted for as well, but that requires a person to scope, design, and purchase. That overall renovation cost is then subtracted from the 70% of ARV to provide a maximum purchase price for this investment.

### Problem Statement
A best-case scenario for a “flip” would be purchasing a like-new home that needs no repairs or updates and can be sold immediately as-is for a profit. This scenario would be a house at 70% or less of the ARV. Therefore, our model should target any homes that are listed at 70% or below the estimated value for that particular property. This problem suggests the use of regression and a possible solution can be using supervised learning to determine an estimate of the resale value of all listed properties. This estimate can then be compared to the actual list price to determine if the property can be acquired for 70% or less of its predicted resale value.

### Solution Statement
In order to estimate the price at which a home could be sold post-repairs, a machine learning model can be fit on real past sales data for the previous 3 months, as reported by Redfin.com. This estimator will then be used to predict at what price a particular home should be able to sell. Then with new listings, comparisons can be made between the newly listed property’s list price and the estimated sale value. Any properties that are listed at or below 70% of the estimated sale value can then be recommended for further inspection as potential “flip” investments.

### Metrics
A new model can be tested by considering how well it would be able to predict past sales prices given the listing information. This means some of the past sales data will need to be withheld from training the model in order to be used during testing the model. The Mean Absolute Error (MAE) can be calculated in order to compare the performance of the newly created estimator model versus that of the benchmark value for that property (price per square foot per zip code) to see which is more accurate at predicting the true sale price of recent sales.

### Algorithm and Techniques
The major work of this project is the data collection and cleaning, rather than exploring many different machine learning algorithms. Since the data gathered and cleaned for this study make a novel dataset to solve a particular problem of interest, the model selected for training these data should be a workhorse algorithm which is accepted as a top performer for price estimation. XGBoost has garnered much attention and respect in the past five years for its superior regression abilities[4][5]. There are countless examples online of price estimators being optimized using the XGBoost algorithm, thus it is a prudent choice for trialing this new dataset. This study makes use of Amazon’s SageMaker built-in algorithm for XGBoost and the supporting SageMaker resources. The transformed samples will be split randomly into training, validation, and testing subsets. 
<br/><br/>
Since this particular study is not a “big data” project, it will be important to reserve enough samples to adequately test the model, but it will also be necessary to give a considerable sized set of samples for training the model. 20% of the randomized and transformed samples will be reserved for testing, while 80% will be used for training and validating the model. Of the 80% of total samples used for training, 30% of samples will be used as validation during training.

### Benchmark Model
Without the aid of machine learning models, list prices are determined by realtors and then “true” house values are determined by an appraiser using a thorough rubric of property characteristics. Appraiser evaluations cost several hundred dollars for each property; thus, a rough evaluation is completed by the realtor using comparable recent sales or other listings in order to determine a proper list price. The most basic way a realtor determines how much to list a house is to consider current listings and recent sales for the immediate neighborhood of the property they are considering. An average price per square foot of house is determined and then multiplied by the square footage of the property they are listing. This is generally a pretty good starting place, and many realtors list properties solely on this number. A more accurate estimate can be determined by considering lot size, number of bedrooms, and number of bathrooms in comparison to recent sales and then adjusting the list price accordingly. However, for the benchmark in this design, it will suffice to consider price per square foot as an average for the zip code within which the home is located.
<br/><br/>
The Jupyter Notebook titled 'Benchmark.ipynb' explores the calculated MAE for this dataset, as found by the benchmark method described above, and this notebook can be found in the supplementary materials. As demonstrated in the Benchmark notebook, the MAE for this benchmark method is $83,028.87.

## Methodology
The material covered in this section can be found in the notebooks titled 'Feature_Engineering.ipynb' and 'XGBoost_Model.ipynb' in the supplementary materials.

## Model Evaluation and Validation
The overall solution exceeded original expectations of the author. The MAE for XGBoost model improves greatly upon the benchmark method, with the MAE for an XGBoost algorithm fit with the parameters above testing in at $32,400.27, a full $50,628.60 improvement from the benchmark Mean Absolute Error for these same samples. In addition to greatly improving the predictive power of what a home’s actual value is (when compared to the benchmark method), the XGBoost model was also able to be used to predict “true” values of properties that are currently listed as “for sale” in this market. 
This model was trained, validated, and tested with randomized samples. However, this is the first implementation of this overall solution. The robustness of this model may be tested with new listings every few months, but due to the time limitations of this project it was not feasible to rigorously test the robustness of this model over several seasons of new real estate sales for this particular geographic region. The decision to use the industry standard XGBoost algorithm was based largely upon this constraint.
<br/><br/>
Using this predicted price, all current listings could be examined against their corresponding prediction to identify any properties that are currently listed as “for sale” that have a list price at or below 70% of the property’s estimated value. This allows for clear identification of available properties which fall into the 70% Rule described in the Problem Definition. Figure 1 shows the properties from this dataset which presented the best potential for a “flip” according to these basic rules.
The overall problem of finding quality potential “flip” properties is improved from manually processing hundreds (initially 604 samples) of active listings to considering only six for this sample set after utilizing this new solution. That is a full order of magnitude improvement! The listing information for each of the recommend six properties is output by 'XGBoost_Model.ipynb'.

### Reflection
This project implemented an XGBoost machine learning algorithm to improve upon the inefficiencies of the “cost per square feet for the neighborhood” benchmark method for pricing homes. The inspiration for this project came from the curiosity for a tool that is capable of digesting real estate listings particular to a geographic region of personal interest, generating an estimated value of each property, and then ultimately recommending current real estate listings that may present an opportunity for a “flip” of the property. As demonstrated by the dramatic (256%) decrease in the MAE of the XGBoost model compared to that of the benchmark model, along with the direct “flip” recommendations shown in Figure 3, this project succeeded at solving the problem statement in its entirety.

### Improvement
While minor improvements in algorithm performance are always possible, this solution does a  major portion of the “heavy lifting” that typically is required for manually digging through real estate listings and depending on human experiential factors to decide if a property “seems” like a good “flip” opportunity. After that human “flagging” of a potential property, individual analysis must be completed. With this new solution, the script is able to determine which properties may be a good “flip” and separate them from the others. The parameter of 70% of expected value could be changed depending on market conditions and risk tolerance factors of the investor. Additionally, it would be nice to extend this project so that it could automatically pull the MLS listings. Unfortunately, I do not have direct read access to an MLS service, but if direct access could be attained, this project could be extended to send a daily email update with any new listings that meet the desired risk tolerance (70% or other value).

 
### References 
[1] https://www.biggerpockets.com/blog/2014-02-14-70-rule-bible <br/> 
[2] https://www.redfin.com/city/17676/CA/Santa-Clarita <br/>
[3] https://www.zillow.com/santa-clarita-ca/home-values/ <br/>
[4] https://towardsdatascience.com/xgboost-the-excalibur-for-everyone-8009bd015f1e <br/>
[5] Chen, Tianqi, and Carlos Guestrin. “Xgboost: A scalable tree boosting system.” Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016. <br/>
[6] Udacity Machine Learning Engineer Nanodegree Course Content
