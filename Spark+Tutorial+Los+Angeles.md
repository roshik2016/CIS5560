
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>


------
<h1 align="center"> CIS5560 Term Project Tutorial </h1>
<h1 align="center"> Predictive Analysis on Income </h1>

------
#### Authors: [Roshik Ganesan](https://www.linkedin.com/in/roshik-ganesan-925143a1); [Kaustubh Padhya ](https://www.linkedin.com/in/kaustubhpadhya);[Mittal Vaghela](https://www.linkedin.com/in/mittal-vaghela-b2811177); [Manali Joshi](https://www.linkedin.com/in/manali-joshi-2a2b9a100)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/18/2017

### Objectives

The aim of this Tutorial is to predict the Income of an employee based on the available features form the dataset by utilizing Machine Learning Algorithms and build accurate models using SparkML

### Cluster creation and specification 

<img alt="California State University, Los Angeles" src="https://github.com/roshik2016/CIS5560/blob/master/Clusterspec.PNG" style="width: 600px;"/>

These are the configuration options for the cluster, <br>
**Spark Version :** Spark 2.1 (Auto-updating, Scala 2.10) <br>
**Memory –** 6GB Memory , 0.88 Cores, 1 DBU <br>
**File System –** DBFS (Data Bricks File System)

### Prepare the Data

First, import the dataset manually using the tables table in the left pane to upload the data, upon uploading the data give the table a name and select the apporpriate datatype for the data.

- Numeric Values - Integer
- Decimal Values - Float
- Values Greater than 65000 - BigInt 
- Charater Values - String

On setting the appropriate datatype click on create table.

 First, Import the libraries you will need and prepare the training and test data.


```python
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor
```

### Import the data from the table 

In this step we are importing the data from the table using sql query and using the count function we find the total number of records in data.


```python
csv =sqlContext.sql("select * from salary2")
maxval = sqlContext.sql("select count(PensionContributions) from salary2")
maxval.show()
```

### Clipping Values from Data 

As we do not have a clip model as seen in azure we need to find the row number for 90th percentile and 20th percentile which is to be clipped.


```python
csv =sqlContext.sql("select * from salary2")
maxval_nine = sqlContext.sql("select count(PensionContributions)*.90 from salary2")
maxval_twen = sqlContext.sql("select count(PensionContributions)*.20 from salary2")
maxval_nine.show()
maxval_twen.show()
```

### Clipping Values from data

As we have found the row number for the clip we do the actual clipping of the data in this step. In this step we have ordered the data based on salary and we have clipped the values of the salary feature. In the following step we will be clipping the values of the Retirement, HealthDental and TotalCompensation and get the data ready for further processing.


```python
#Ordering the column PensionContribution and providing row numbers to data
rownumber = sqlContext.sql("select ROW_NUMBER() over (ORDER BY PensionContributions) AS Row, PensionContributions from salary2")
rownumber.createOrReplaceTempView("res1")
#Finding the values corresponding to the row number which was found in the earlier step
val_nineper = sqlContext.sql("select PensionContributions,Row from res1 where Row = '277110' ")
val_twenper = sqlContext.sql("select PensionContributions,Row from res1 where Row = '61580' ")
val_twenper.show()
val_nineper.show()
```


```python
#Ordering the column MedicalDentalVision and providing row numbers to data
rownumber = sqlContext.sql("select ROW_NUMBER() over (ORDER BY MedicalDentalVision) AS Row, MedicalDentalVision from salary2")
rownumber.createOrReplaceTempView("res1")
#Finding the values corresponding to the row number which was found in the earlier step
val_nineper = sqlContext.sql("select MedicalDentalVision,Row from res1 where Row = '277110' ")
val_twenper = sqlContext.sql("select MedicalDentalVision,Row from res1 where Row = '61580' ")
val_twenper.show()
val_nineper.show()
```


```python
#Ordering the column TotalCompensation and providing row numbers to data
rownumber = sqlContext.sql("select ROW_NUMBER() over (ORDER BY TotalCompensation) AS Row, TotalCompensation from salary2")
rownumber.createOrReplaceTempView("res1")
#Finding the values corresponding to the row number which was found in the earlier step
val_nineper = sqlContext.sql("select TotalCompensation,Row from res1 where Row = '277110' ")
val_twenper = sqlContext.sql("select TotalCompensation,Row from res1 where Row = '61580' ")
val_twenper.show()
val_nineper.show()
```


```python
#Ordering the column LTDLifeMedicalTax and providing row numbers to data
rownumber = sqlContext.sql("select ROW_NUMBER() over (ORDER BY LTDLifeMedicalTax) AS Row, LTDLifeMedicalTax from salary2")
rownumber.createOrReplaceTempView("res1")
#Finding the values corresponding to the row number which was found in the earlier step
val_nineper = sqlContext.sql("select LTDLifeMedicalTax,Row from res1 where Row = '277110' ")
val_twenper = sqlContext.sql("select LTDLifeMedicalTax,Row from res1 where Row = '61580' ")
val_twenper.show()
val_nineper.show()
```

### Clipping Values from the data

Here we pass the values of the feature that we have found for the corresponding row numbers.


```python
#Casting the data to a double datatype as there are a few inconsistant records in the data which would create an issue when the model is trained
csv1 = sqlContext.sql(" select cast(PensionContributions as double),cast(MedicalDentalVision as double),cast(TotalCompensation as double),cast(LTDLifeMedicalTax as double) from salary2")
#Dropping null values
csv1= csv1.dropna()
#Clipping the outliers from the data
data = csv1.select("PensionContributions","MedicalDentalVision","LTDLifeMedicalTax", col("TotalCompensation").alias("label")).where(col("PensionContributions") >= ((6321.92))).where (col("PensionContributions") <= (24482.66)).where(col("MedicalDentalVision") >= (6019.92)).where (col("MedicalDentalVision") <= (18807.05)).where(col("TotalCompensation") >= (57087.26)).where (col("TotalCompensation") <= (177859.19)).where (col("LTDLifeMedicalTax") >= (540.22)).where (col("LTDLifeMedicalTax") <= (2303.66))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
```

### Algorithm 1 - Decission Tree Algorithm

A decission Tress algorithim is used for the first model to train the data for prediction

### Create a Vector Assembler 

Now we create a vector assemble which would assemble all the selected feature and prepare it for pipeline


```python
assembler = VectorAssembler(inputCols = ["PensionContributions","MedicalDentalVision","LTDLifeMedicalTax"], outputCol="features")
dt = DecisionTreeRegressor(labelCol ="label",featuresCol="features")
```

### Tune Parameters
You can tune parameters to find the best model for your data. To do this you can use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.

### Why cross-validation: 
Using one training set and one validation set could still end up over fitting your model that might not always produce the optimal model with the optimal parameters hence a cross validator is being used in this scenario


```python
paramGrid = ParamGridBuilder()\
  .addGrid(dt.maxDepth, [5, 10])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=dt.getLabelCol(), predictionCol=dt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

### Define the Pipeline
Now define a pipeline that creates a feature vector and trains a regression model


```python
# Define the pipeline
pipeline = Pipeline(stages=[assembler, cv])
pipelineModel = pipeline.fit(train)
```

### Test the Model
Now you're ready to apply the model to the test data.


```python
predictions = pipelineModel.transform(test)
```


```python
predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(100)
```

### Plotting Predicted and Actual Values

Based on the predection from the transform model the actual values and the predicted values are ploted in a scatter plot by creating a temporary view and we are close to acheving a diagonal line.


```python
predicted.createOrReplaceTempView("regressionPredictions")
```


```python
# Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
# convert to pandas and plot
display(dataPred)
```

### RMSE Analysis

Beased on the RMSE (Root Mean Squared Error) this Model can be evaluated.


```python
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print "Root Mean Square Error (RMSE) for Decession Tree Model:", rmse
```

### Algorithm 2 - GBT Regressor (Gradient Booster Tree Regression)

The data is put to learn using a different machine learning algorithm (GBT) so that a comparison could be made and the best algorithm could be analyzed. The features are first assembled and then the model is trained and evaluated as done previously.


```python
assembler = VectorAssembler(inputCols = ["PensionContributions","MedicalDentalVision","LTDLifeMedicalTax"], outputCol="features")
gbt = GBTRegressor(labelCol="label")
```

### Tune Parameters
You can tune parameters to find the best model for your data. To do this you can use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.


```python
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

### Define the Pipeline
Now define a pipeline that creates a feature vector and trains a regression model


```python
pipeline = Pipeline(stages=[assembler, cv])
pipelineModel = pipeline.fit(train)
```


```python
predictions = pipelineModel.transform(test)
```


```python
predicted = predictions.select("features", "prediction", "trueLabel")
display(predicted)
```

### Plotting Predicted and Actual Values

Based on the predection from the transform model the actual values and the predicted values are ploted in a scatter plot by creating a temporary view and we are close to acheving a diagonal line.


```python
predicted.createOrReplaceTempView("regressionPredictions")
```


```python
# Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
# convert to pandas and plot
display(dataPred)
```

### RMSE Analysis

Beased on the RMSE (Root Mean Squared Error) this algorithm can be evaluated.


```python
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print "Root Mean Square Error (RMSE) for GBT Regression :", rmse
```
